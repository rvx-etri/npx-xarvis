from pathlib import *
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

import tonic

from collections import Counter
from requests import get
from npx_cfg_parser import *
from npx_define import *

from npx_speechcommands import *
from sklearn.model_selection import GroupShuffleSplit

def download(url: str, root: Path, file_name=None):
    if not file_name:
        file_name = url.split('/')[-1]

    path = root / file_name
    with open(path, "wb") as file:
        response = get(url)
        file.write(response.content)


def intstr_to_tuple(intstr):
    intstr_list = intstr.split(',')
    int_list = []
    for i in range(len(intstr_list)):
        int_list.append(int(intstr_list[i]))
    return tuple(int_list)


def build_image_transform_pair(preprocess_info, pre_ops, post_ops):
    """Build (train_transform, eval_transform, augmentation) for an image dataset.

    Augmentation is opt-in via `[preprocess] augmentation=`; absent it defaults to
    'none' and both transforms are the same object, reproducing the previous
    behaviour exactly. `pre_ops` are the transforms that must run before
    augmentation (Resize / Grayscale); `post_ops` are ToTensor + Normalize, which
    must run after it because the augmentation ops operate on PIL images.

    Individual knobs may be overridden per app with dotted `augment.*` keys, in the
    same style as the existing `mel_spectrogram.*` options.
    """
    augmentation = str(preprocess_info.setdefault(
        'augmentation', 'none')).lower()
    eval_transform = transforms.Compose(list(pre_ops) + list(post_ops))
    if augmentation == 'none':
        return eval_transform, eval_transform, augmentation
    assert augmentation == 'affine', f'unknown augmentation "{augmentation}" (expected none|affine)'

    def opt(name, default):
        return preprocess_info.setdefault(f'augment.{name}', default)
    degrees = float(opt('degrees', 10))
    translate = float(opt('translate', 0.1))
    scale = opt('scale', (0.9, 1.1))
    brightness = float(opt('brightness', 0.3))
    contrast = float(opt('contrast', 0.3))
    saturation = float(opt('saturation', 0.2))
    hflip = float(opt('hflip', 0.0))
    vflip = float(opt('vflip', 0.0))
    assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        f'augment.scale must be two values, e.g. "augment.scale=0.9,1.1" (got {scale!r})'
    scale = tuple(float(s) for s in scale)

    aug_ops = []
    if degrees or translate or scale != (1.0, 1.0):
        aug_ops.append(transforms.RandomAffine(
            degrees=degrees,
            translate=((translate, translate) if translate else None),
            scale=(scale if scale != (1.0, 1.0) else None)))
    if brightness or contrast or saturation:
        aug_ops.append(transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation))
    # Flips are OFF by default and must stay off for chirality-sensitive data:
    # flipping a traffic sign (gtsrb) or a digit/letter (mnist/kmnist) changes or
    # destroys its class. Enable per app only where it is meaningful (e.g. cifar10).
    if hflip:
        aug_ops.append(transforms.RandomHorizontalFlip(p=hflip))
    if vflip:
        aug_ops.append(transforms.RandomVerticalFlip(p=vflip))

    train_transform = transforms.Compose(
        list(pre_ops) + aug_ops + list(post_ops))
    return train_transform, eval_transform, augmentation


SPLIT_METHOD_NAMES = ('stratified_group', 'stratified', 'random')


class NpxDataManager():
    def _setup_image_datasets(self, preprocess_info, pre_ops, post_ops, make_train, make_test):
        """Wire up train/test datasets for an image dataset, honouring augmentation.

        `make_train`/`make_test` build the dataset given a transform. When
        augmentation is on, a second train dataset is built with the clean transform
        so the validation fold can be indexed from it (random_split returns Subsets
        of one dataset object, which would otherwise share the augmenting transform).
        """
        train_transform, eval_transform, augmentation = build_image_transform_pair(
            preprocess_info, pre_ops, post_ops)
        self.augmentation = augmentation
        # Kept so training can switch augmentation on/off between epochs (see
        # set_augmentation): a deep SNN's silent state is absorbing, and starting
        # straight into augmented data can drive it there before it has found any
        # class signal.
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        dataset_train_and_val = make_train(train_transform)
        self.train_dataset_object = dataset_train_and_val
        self.dataset_test = make_test(eval_transform)
        if augmentation != 'none':
            self.dataset_clean_source = make_train(eval_transform)
        return dataset_train_and_val

    @staticmethod
    def _extract_labels_and_groups(dataset):
        """Return (labels, groups) as arrays aligned with the dataset order.

        Only path-backed image datasets are supported: the group id is read from
        the file name, which for gtsrb is `<track>_<frame>.ppm`. A track is one
        physical sign photographed ~30 times, and it never spans two classes, so
        keeping a track whole also keeps the split stratified. Labels are taken
        from the sample list rather than by iterating the dataset, which would
        decode every image just to read its label.
        """
        samples = getattr(dataset, '_samples', None)
        if samples is None:
            samples = getattr(dataset, 'samples', None)
        assert samples is not None, \
            f'split_method=stratified_group needs a path-backed dataset ' \
            f'(no _samples/samples on {type(dataset).__name__})'
        labels = []
        groups = []
        for path, label in samples:
            track = Path(path).stem.split('_')[0]
            labels.append(label)
            groups.append(f'{label}/{track}')
        return np.asarray(labels), np.asarray(groups)

    @staticmethod
    def _reject_augmentation(preprocess_info, dataset_name):
        assert str(preprocess_info.get('augmentation', 'none')).lower() == 'none', \
            f'augmentation is not supported for the {dataset_name} input pipeline ' \
            f'(no PIL image stage); remove `augmentation=` from [preprocess]'

    def __init__(self, npx_define: NpxDefine, dataset_path: Path, kfold: int = None):

        self.name = npx_define.cfg_parser.preprocess_info['input']
        is_open_dataset = False
        if self.name.endswith('_opendataset'):
            self.name = self.name[:-len('_opendataset')]
            opendatatset_list = ['mnist', 'kmnist', 'fmnist',
                                 'cifar10', 'gtsrb', 'dvsgesture', 'speechcommands']
            assert self.name in opendatatset_list, self.name
            is_open_dataset = True
        elif self.name.endswith('_dataset'):
            self.name = self.name[:-len('_dataset')]

        self.download_path = dataset_path / self.name
        if is_open_dataset:
            self.download_path.mkdir(parents=True, exist_ok=True)
        else:
            assert self.download_path.is_dir(), self.download_path

        if kfold == None:
            kfold = 5
        assert kfold >= 4, kfold
        self.kfold = kfold
        self.split_method = npx_define.cfg_parser.train_info.setdefault(
            'split_method', None)

        if self.name == 'mnist':
            if not self.split_method:
                self.split_method = 'stratified'
            self.raw_data_format = DataFormat.MATRIX3D
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
            # self.resize = intstr_to_tuple(value)
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                # transforms.Resize((14, 14)),
                # transforms.Resize(self.resize),
                [transforms.Grayscale()],
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))],
                lambda t: datasets.MNIST(
                    root=self.download_path, train=True, download=True, transform=t),
                lambda t: datasets.MNIST(root=self.download_path, train=False, download=True, transform=t))
        elif self.name == 'kmnist':
            if not self.split_method:
                self.split_method = 'stratified'
            self.raw_data_format = DataFormat.MATRIX3D
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
            # self.resize = intstr_to_tuple(value)
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                # transforms.Resize((14, 14)),
                # transforms.Resize(self.resize),
                [transforms.Grayscale()],
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))],
                lambda t: datasets.KMNIST(
                    root=self.download_path, train=True, download=True, transform=t),
                lambda t: datasets.KMNIST(root=self.download_path, train=False, download=True, transform=t))
        elif self.name == 'fmnist':
            if not self.split_method:
                self.split_method = 'stratified'
            self.raw_data_format = DataFormat.MATRIX3D
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
            # self.resize = intstr_to_tuple(value)
            download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz',
                     self.download_path / 'FashionMNIST/raw')
            download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz',
                     self.download_path / 'FashionMNIST/raw')
            download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz',
                     self.download_path / 'FashionMNIST/raw')
            download('https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz',
                     self.download_path / 'FashionMNIST/raw')
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                # transforms.Resize((14, 14)),
                # transforms.Resize(self.resize),
                [transforms.Grayscale()],
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))],
                lambda t: datasets.FashionMNIST(
                    root=self.download_path, train=True, download=True, transform=t),
                lambda t: datasets.FashionMNIST(root=self.download_path, train=False, download=True, transform=t))
        elif self.name == 'cifar10':
            if not self.split_method:
                self.split_method = 'stratified'
            self.raw_data_format = DataFormat.MATRIX3D
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '32,32')
            # self.resize = intstr_to_tuple(value)
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                # transforms.Resize((32, 32)),
                # transforms.Resize(self.resize),
                [],
                [transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))],
                lambda t: datasets.CIFAR10(
                    root=self.download_path, train=True, download=True, transform=t),
                lambda t: datasets.CIFAR10(root=self.download_path, train=False, download=True, transform=t))
        elif self.name == 'gtsrb':
            if not self.split_method:
                self.split_method = 'stratified_group'
            self.raw_data_format = DataFormat.MATRIX3D
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # Resize to whatever [global] input_size asks for (default 32x32, the
            # previous hardcoded value). Doing it here rather than leaving it at
            # 32x32 matters: npx_trainer interpolates the batch up to input_size
            # if they differ, so a cfg asking for 48x48 would otherwise be fed a
            # 32x32 resize upsampled to 48x48 -- more compute, no more detail.
            self.resize = tuple(npx_define.cfg_parser.global_info.get('input_size', (32, 32)))
            assert len(self.resize) == 2, self.resize
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                [transforms.Resize(self.resize)],
                [transforms.ToTensor(), transforms.Normalize((0, 0, 0), (1, 1, 1))],
                lambda t: datasets.GTSRB(
                    root=self.download_path, split='train', download=True, transform=t),
                lambda t: datasets.GTSRB(root=self.download_path, split='test', download=True, transform=t))
        elif self.name == 'dvsgesture':
            if not self.split_method:
                self.split_method = 'stratified'
            self._reject_augmentation(
                npx_define.cfg_parser.preprocess_info, 'dvsgesture')
            self.raw_data_format = DataFormat.DVS
            self.data_format = DataFormat.MATRIX4D
            self.sensor_size = tonic.datasets.DVSGesture.sensor_size
            # denoise_transform = tonic.transforms.Denoise(filter_time=10000)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '128,128')
            # self.resize = intstr_to_tuple(value)
            # self.resize = (2,) + self.resize
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 32)
            frame_transform = tonic.transforms.ToFrame(
                sensor_size=self.sensor_size, n_time_bins=self.timesteps)
            all_transform = transforms.Compose([
                frame_transform])
            train_set = tonic.datasets.DVSGesture(
                save_to=self.download_path, transform=all_transform, train=True)
            test_set = tonic.datasets.DVSGesture(
                save_to=self.download_path, transform=all_transform, train=False)
            dataset_train_and_val = tonic.DiskCachedDataset(
                train_set, cache_path=self.download_path/'cache/dvsgesture/train')
            self.dataset_test = tonic.DiskCachedDataset(
                test_set, cache_path=self.download_path/'cache/dvsgesture/test')
            self.dataset_test_raw = tonic.datasets.DVSGesture(
                save_to=self.download_path, train=False)
        elif self.name == 'speechcommands':
            if not self.split_method:
                self.split_method = 'stratified'
            self._reject_augmentation(
                npx_define.cfg_parser.preprocess_info, 'speechcommands')
            self.raw_data_format = DataFormat.WAVEFORM
            self.data_format = DataFormat.MATRIX3D
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            self.feature = npx_define.cfg_parser.preprocess_info.setdefault(
                'feature', None)
            self.transform = None
            self.sample_rate = 16000
            if self.feature == 'mel_spectrogram':
                self.num_samples = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.num_samples', 16000)
                self.sample_rate = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.sample_rate', 16000)
                self.n_fft = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.n_fft', 512)
                self.win_length = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.win_length', 400)
                self.hop_length = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.hop_length', 160)
                self.n_mels = npx_define.cfg_parser.preprocess_info.setdefault(
                    'mel_spectrogram.n_mels', 40)
                self.transform = NpxMelSpectrogram(
                    num_samples=self.num_samples,
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
            self.output_classes = int( npx_define.cfg_parser.global_info.get("output_classes", 3))

            train_dataset = NpxSpeechCommandsPreprocess(
                root=self.download_path, subset='training', output_classes=self.output_classes,
                transform=self.transform, target_sr=self.sample_rate)
            val_dataset = NpxSpeechCommandsPreprocess(
                root=self.download_path, subset='validation', output_classes=self.output_classes,
                transform=self.transform, target_sr=self.sample_rate)
            dataset_train_and_val = train_dataset + val_dataset
            self.dataset_test = NpxSpeechCommandsPreprocess(
                root=self.download_path, subset='testing', output_classes=self.output_classes,
                transform=self.transform, target_sr=self.sample_rate)
            self.dataset_test_raw = NpxSpeechCommandsPreprocess(
                root=self.download_path, subset='testing', output_classes=self.output_classes,
                transform=None, target_sr=self.sample_rate)
        else:
            print(f'Custom Dataset: {self.name}')
            if not self.split_method:
                self.split_method = 'stratified'
            assert self.download_path.is_dir(
            ), f"Dataset does not exist in the path: {self.download_path}"
            self.input_type = npx_define.cfg_parser.preprocess_info.setdefault(
                'input_type', 'image')
            if self.input_type == 'waveform':
                self.raw_data_format = DataFormat.WAVEFORM
            else:
                self.raw_data_format = DataFormat.MATRIX3D
            # if self.input_type == 'waveform' ??? dododo
            self.data_format = DataFormat.MATRIX3D
            self.step_generation = npx_define.cfg_parser.preprocess_info.setdefault(
                'step_generation', 'direct')
            self.timesteps = npx_define.cfg_parser.preprocess_info.setdefault(
                'timesteps', 4)
            # value = NpxCfgParser.find_option_value(npx_define.cfg_parser.preprocess_info, 'resize', '14,14')
            # self.resize = intstr_to_tuple(value)
            dataset_train_and_val = self._setup_image_datasets(
                npx_define.cfg_parser.preprocess_info,
                # transforms.Resize(self.resize),
                [transforms.Grayscale()],
                [transforms.ToTensor(), transforms.Normalize((0,), (1,))],
                # transforms.PILToTensor(),
                # transforms.Lambda(lambda t: t.to(torch.float32))])
                lambda t: datasets.MNIST(
                    root=self.download_path, train=True, download=False, transform=t),
                lambda t: datasets.MNIST(root=self.download_path, train=False, download=False, transform=t))

        if self.split_method == 'stratified_group':
            # Correlated samples (a gtsrb track: the same physical sign shot ~30
            # times) must land in one fold only - split them at sample level and
            # a validation sign has near-duplicates in training, which inflates
            # the validation score. GroupShuffleSplit is run per label rather
            # than over the whole set so the folds stay label-balanced as well;
            # that is safe because a group never spans two labels.
            labels, groups = self._extract_labels_and_groups(
                dataset_train_and_val)
            fold_index_lists = [[] for _ in range(kfold)]
            for label in np.unique(labels):
                remaining = np.flatnonzero(labels == label)
                num_groups = len(np.unique(groups[remaining]))
                assert num_groups >= kfold, \
                    f'label {label} has {num_groups} group(s), too few for {kfold} folds'
                # One GroupShuffleSplit yields two parts, so peel off one fold at
                # a time: 1/kfold of what is left, then 1/(kfold-1), and so on.
                for fold in range(kfold - 1):
                    splitter = GroupShuffleSplit(
                        n_splits=1, test_size=1.0/(kfold - fold), random_state=fold)
                    rest, held_out = next(splitter.split(
                        remaining, labels[remaining], groups[remaining]))
                    fold_index_lists[fold].extend(remaining[held_out].tolist())
                    remaining = remaining[rest]
                fold_index_lists[-1].extend(remaining.tolist())
            # Subsets of the dataset object itself (not of per-label lists), so
            # the fold indices can be mirrored onto a clean-transform twin below.
            self.dataset_both_list = [
                torch.utils.data.Subset(dataset_train_and_val, sorted(index_list))
                for index_list in fold_index_lists]
        elif self.split_method == 'stratified':
            labeled_dataset_dict = {}
            for data, label in dataset_train_and_val:
                labeled_dataset = labeled_dataset_dict.get(label)
                if labeled_dataset:
                    labeled_dataset.append((data, label))
                else:
                    labeled_dataset_dict[label] = [(data, label)]
            self.dataset_both_list = None
            for labeled_dataset in labeled_dataset_dict.values():
                chunk_size = int(len(labeled_dataset)/self.kfold)
                chunk_size_list = []
                for i in range(0, kfold-1):
                    chunk_size_list.append(chunk_size)
                last_chunk_size = len(labeled_dataset) - (chunk_size*(kfold-1))
                chunk_size_list.append(last_chunk_size)
                splited_labeled_dataset = torch.utils.data.random_split(
                    labeled_dataset, chunk_size_list)
                if not self.dataset_both_list:
                    self.dataset_both_list = splited_labeled_dataset
                else:
                    for i in range(len(self.dataset_both_list)):
                        self.dataset_both_list[i] += splited_labeled_dataset[i]
        elif self.split_method == 'random':
            chunk_size = int(len(dataset_train_and_val)/self.kfold)
            chunk_size_list = []
            for i in range(0, kfold-1):
                chunk_size_list.append(chunk_size)
            last_chunk_size = len(dataset_train_and_val) - \
                (chunk_size*(kfold-1))
            chunk_size_list.append(last_chunk_size)
            self.dataset_both_list = torch.utils.data.random_split(
                dataset_train_and_val, chunk_size_list)
        else:
            assert 0, self.split_method

        # Validation folds normally reuse the training dataset object. When an
        # augmenting train transform is in use, mirror the same fold indices onto a
        # clean-transform twin so validation is measured without augmentation.
        clean_source = getattr(self, 'dataset_clean_source', None)
        if clean_source is None:
            self.dataset_val_list = self.dataset_both_list
        else:
            # 'stratified' splits per-label lists, so its Subset indices do not
            # address the underlying dataset and could not be mirrored.
            assert self.split_method != 'stratified', \
                'augmentation + split_method=stratified is unsupported'
            self.dataset_val_list = [torch.utils.data.Subset(clean_source, subset.indices)
                                     for subset in self.dataset_both_list]

    def set_augmentation(self, enabled: bool):
        """Turn the augmenting train transform on or off between epochs.

        No-op unless the cfg opted into augmentation. The transform is swapped on
        the dataset object shared by every training Subset; loaders are built with
        persistent_workers=False so freshly forked workers pickle the current state.
        Validation/test read from a separate clean-transform dataset and are never
        affected.
        """
        if getattr(self, 'augmentation', 'none') == 'none':
            return
        self.train_dataset_object.transform = self.train_transform if enabled else self.eval_transform

    def setup_loader(self, kfold_index: int, batch_size: int = 100):
        assert kfold_index <= len(self.dataset_both_list), len(
            self.dataset_both_list)
        self.dataset_val = self.dataset_val_list[kfold_index]
        dataset_train_subset_list = self.dataset_both_list[:kfold_index] + \
            self.dataset_both_list[kfold_index+1:]
        self.dataset_train = ConcatDataset(dataset_train_subset_list)
        print('The number of train dataset:', len(self.dataset_train))
        print('The number of valid dataset:', len(self.dataset_val))
        if self.name == 'dvsgesture':
            collate_fn = tonic.collation.PadTensors(batch_first=False)
        else:
            collate_fn = None
        # PIL-side augmentation is expensive and runs inline with training when
        # num_workers==0, so give the training loader workers only in that case.
        # Batch composition is unchanged (the sampler still runs in the main
        # process), so non-augmented apps keep their exact previous behaviour.
        num_workers = 8 if getattr(
            self, 'augmentation', 'none') != 'none' else 0
        self.loader_list = (DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers),
                            DataLoader(self.dataset_val, batch_size=batch_size,
                                       shuffle=True, drop_last=True, collate_fn=collate_fn),
                            DataLoader(self.dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn))

    @property
    def train_loader(self):
        return self.loader_list[0]

    @property
    def val_loader(self):
        return self.loader_list[1]

    @property
    def test_loader(self):
        return self.loader_list[2]

    @property
    def input_size(self):
        return self.dataset_test[0][0].shape

    @property
    def num_classes(self):
        return len(dict(Counter(sample_tup[1] for sample_tup in self.dataset_test)))
