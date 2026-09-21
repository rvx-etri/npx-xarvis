"""Download and prepare the open datasets ahead of training.

Each dataset goes to <dataset>/<name>/, the path npx_data_manager reads for
input=<name>_opendataset, so training finds it and downloads nothing.

ecg, gas and har have no loader of their own: they are converted to the MNIST format
(<dataset>/<name>/MNIST/raw/{train,t10k}-{images-idx3,labels-idx1}-ubyte).
  ecg  MIT-BIH arrhythmia, one beat (256 samples around the R peak) as a 16x16 image,
       5 AAMI classes N/S/V/F/Q.
       -ecg_protocol intra (default): 10 % of the beats of every record held out
                     inter: de Chazal DS1 train / DS2 test, paced records excluded
  gas  UCI gas sensor array drift, 16 sensors x 8 features as a 16x8 image, 6 gases.
       -gas_protocol random (default): 20 % held out per gas
                     drift: batches 1-6 train, 7-10 test
  har  UCI human activity recognition, 9 inertial channels x 128 samples as a 9x128 image,
       6 activities, the dataset's own subject-wise test split.
They are converted once; -force converts again (needed after changing a protocol).

Usage:
  python3 npx_opendataset_prepare.py -dataset ../dataset -i all
  python3 npx_opendataset_prepare.py -dataset ../dataset -i ecg gas
"""

import argparse
import hashlib
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

OPEN_DATASETS = ('mnist', 'kmnist', 'fmnist', 'cifar10', 'gtsrb', 'dvsgesture', 'speechcommands', 'ecg', 'gas', 'har')
IDX_FILES = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


def write_idx(images, labels, raw_dir: Path, prefix: str):
  n, h, w = images.shape
  with open(raw_dir / f'{prefix}-images-idx3-ubyte', 'wb') as f:
    f.write(struct.pack('>IIII', 2051, n, h, w))
    f.write(images.astype(np.uint8).tobytes())
  with open(raw_dir / f'{prefix}-labels-idx1-ubyte', 'wb') as f:
    f.write(struct.pack('>II', 2049, n))
    f.write(labels.astype(np.uint8).tobytes())


def is_converted(root: Path):
  raw_dir = root / 'MNIST' / 'raw'
  return all((raw_dir / name).is_file() for name in IDX_FILES)


def prepare_mnist(root: Path):
  from torchvision import datasets
  for train in (True, False):
    datasets.MNIST(root=root, train=train, download=True)


def prepare_kmnist(root: Path):
  from torchvision import datasets
  for train in (True, False):
    datasets.KMNIST(root=root, train=train, download=True)


FMNIST_URL = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion'


def prepare_fmnist(root: Path):
  from torchvision import datasets
  raw_dir = root / 'FashionMNIST' / 'raw'
  raw_dir.mkdir(parents=True, exist_ok=True)
  for name in IDX_FILES:
    archive = raw_dir / f'{name}.gz'
    if not (raw_dir / name).is_file() and not archive.is_file():
      print('downloading', f'{FMNIST_URL}/{name}.gz')
      urllib.request.urlretrieve(f'{FMNIST_URL}/{name}.gz', archive)
  for train in (True, False):
    datasets.FashionMNIST(root=root, train=train, download=True)


def prepare_cifar10(root: Path):
  from torchvision import datasets
  for train in (True, False):
    datasets.CIFAR10(root=root, train=train, download=True)


def prepare_gtsrb(root: Path):
  from torchvision import datasets
  for split in ('train', 'test'):
    datasets.GTSRB(root=root, split=split, download=True)


def prepare_dvsgesture(root: Path):
  import tonic
  for train in (True, False):
    tonic.datasets.DVSGesture(save_to=root, train=train)


def prepare_speechcommands(root: Path):
  from torchaudio.datasets import SPEECHCOMMANDS
  SPEECHCOMMANDS(root, download=True)


ECG_URL = 'https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip'
ECG_BEFORE, ECG_AFTER = 90, 166
ECG_SIDE = 16

# MIT annotation codes -> AAMI class
ECG_AAMI = {1: 0, 2: 0, 3: 0, 16: 0, 11: 0,     # N L R e j
            8: 1, 4: 1, 7: 1, 9: 1,             # A a J S
            5: 2, 10: 2,                        # V E
            6: 3,                               # F
            12: 4, 38: 4, 13: 4}                # / f Q
ECG_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205,
           207, 208, 209, 215, 220, 223, 230]
ECG_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219,
           221, 222, 228, 231, 232, 233, 234]


def fetch_ecg(download_dir: Path):
  archive = download_dir / 'mitdb.zip'
  if not any(download_dir.rglob('100.hea')):
    download_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
      print('downloading', ECG_URL)
      urllib.request.urlretrieve(ECG_URL, archive)
    print('extracting', archive)
    with zipfile.ZipFile(archive) as z:
      z.extractall(download_dir)
  return next(download_dir.rglob('100.hea')).parent


def read_ecg_signal(record_dir: Path, record: str):
  header = (record_dir / f'{record}.hea').read_text().split('\n')
  nsig = int(header[0].split()[1])
  assert nsig == 2 and header[1].split()[1] == '212', header[:2]
  raw = np.fromfile(record_dir / f'{record}.dat', dtype=np.uint8)
  raw = raw[: len(raw) // 3 * 3].reshape(-1, 3).astype(np.int32)
  first = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
  return np.where(first > 2047, first - 4096, first)


def read_ecg_beats(record_dir: Path, record: str):
  data = np.fromfile(record_dir / f'{record}.atr', dtype='<u2')
  out, t, i = [], 0, 0
  while i < len(data):
    word = int(data[i]); code, interval = word >> 10, word & 0x3FF
    i += 1
    if code == 0 and interval == 0:
      break
    if code == 59:
      t += (int(data[i]) << 16) | int(data[i + 1])
      i += 2
    elif code == 63:
      i += (interval + 1) // 2
    elif code in (60, 61, 62):
      pass
    else:
      t += interval
      out.append((t, code))
  return out


def prepare_ecg(root: Path, force: bool = False, protocol: str = 'intra', test_percent: int = 10):
  if is_converted(root) and not force:
    return
  record_dir = fetch_ecg(root / 'download')
  records = sorted(p.stem for p in record_dir.glob('*.hea'))
  if protocol == 'inter':
    part_of = {str(r): 'train' for r in ECG_DS1}
    part_of.update({str(r): 't10k' for r in ECG_DS2})
    records = [r for r in records if r in part_of]

  split = {'train': ([], []), 't10k': ([], [])}
  for record in records:
    signal = read_ecg_signal(record_dir, record)
    for sample, code in read_ecg_beats(record_dir, record):
      label = ECG_AAMI.get(code)
      if label is None or sample < ECG_BEFORE or sample + ECG_AFTER > len(signal):
        continue
      beat = signal[sample - ECG_BEFORE: sample + ECG_AFTER].astype(np.float32)
      span = beat.max() - beat.min()
      beat = (beat - beat.min()) / span * 255.0 if span > 0 else np.zeros_like(beat)
      if protocol == 'inter':
        part = part_of[record]
      else:
        digest = hashlib.sha1(f'{record}:{sample}'.encode()).digest()
        part = 't10k' if int.from_bytes(digest[:4], 'big') % 100 < test_percent else 'train'
      split[part][0].append(np.round(beat).astype(np.uint8).reshape(ECG_SIDE, ECG_SIDE))
      split[part][1].append(label)

  raw_dir = root / 'MNIST' / 'raw'
  raw_dir.mkdir(parents=True, exist_ok=True)
  for part, (images, labels) in split.items():
    labels = np.asarray(labels)
    write_idx(np.stack(images), labels, raw_dir, part)
    counts = np.bincount(labels, minlength=5)
    print(f'ecg {part}: {len(labels)} beats  N/S/V/F/Q = {"/".join(str(c) for c in counts)}')
  print(f'ecg protocol={protocol}; wrote', raw_dir)


GAS_URL = 'https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip'


def fetch_gas(download_dir: Path):
  download_dir.mkdir(parents=True, exist_ok=True)
  path = download_dir / 'gas.zip'
  if not path.is_file():
    print('downloading', GAS_URL)
    urllib.request.urlretrieve(GAS_URL, path)
  return path


def read_gas_batch(text):
  xs, ys = [], []
  for line in text.splitlines():
    parts = line.split()
    if not parts:
      continue
    ys.append(int(parts[0]) - 1)
    row = np.zeros(128, dtype=np.float32)
    for item in parts[1:]:
      idx, value = item.split(':')
      row[int(idx) - 1] = float(value)
    xs.append(row)
  return np.stack(xs), np.array(ys)


def prepare_gas(root: Path, force: bool = False, protocol: str = 'random', seed: int = 0):
  if is_converted(root) and not force:
    return
  archive = fetch_gas(root / 'download')
  batches = {}
  with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
      if name.endswith('.dat'):
        b = int(Path(name).stem.replace('batch', ''))
        batches[b] = read_gas_batch(z.read(name).decode())
  if protocol == 'drift':
    train = [batches[b] for b in range(1, 7)]
    test = [batches[b] for b in range(7, 11)]
    train_x, train_y = np.concatenate([t[0] for t in train]), np.concatenate([t[1] for t in train])
    test_x, test_y = np.concatenate([t[0] for t in test]), np.concatenate([t[1] for t in test])
  else:
    x = np.concatenate([batches[b][0] for b in sorted(batches)])
    y = np.concatenate([batches[b][1] for b in sorted(batches)])
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in range(6):
      idx = np.flatnonzero(y == c)
      test_idx += rng.choice(idx, size=len(idx) // 5, replace=False).tolist()
    mask = np.zeros(len(y), dtype=bool)
    mask[test_idx] = True
    train_x, train_y, test_x, test_y = x[~mask], y[~mask], x[mask], y[mask]

  low = np.percentile(train_x, 0.5, axis=0)
  high = np.percentile(train_x, 99.5, axis=0)
  def scale(v):
    img = np.clip(np.round((v - low) / np.maximum(high - low, 1e-6) * 255.0), 0, 255).astype(np.uint8)
    return img.reshape(-1, 16, 8)

  raw_dir = root / 'MNIST' / 'raw'
  raw_dir.mkdir(parents=True, exist_ok=True)
  for prefix, v, lab in (('train', train_x, train_y), ('t10k', test_x, test_y)):
    write_idx(scale(v), lab, raw_dir, prefix)
    print(f'gas {prefix}: {len(lab)} measurements, 16x8, per class {np.bincount(lab, minlength=6).tolist()}')
  print(f'gas protocol={protocol}; wrote', raw_dir)


HAR_URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
HAR_CHANNELS = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
                'total_acc_x', 'total_acc_y', 'total_acc_z']


def fetch_har(download_dir: Path):
  found = list(download_dir.rglob('UCI HAR Dataset'))
  if not found:
    download_dir.mkdir(parents=True, exist_ok=True)
    outer = download_dir / 'uci_har.zip'
    if not outer.is_file():
      print('downloading', HAR_URL)
      urllib.request.urlretrieve(HAR_URL, outer)
    print('extracting', outer)
    with zipfile.ZipFile(outer) as z:
      z.extractall(download_dir)
    inner = download_dir / 'UCI HAR Dataset.zip'
    if inner.is_file():
      with zipfile.ZipFile(inner) as z:
        z.extractall(download_dir)
    found = list(download_dir.rglob('UCI HAR Dataset'))
  return [p for p in found if p.is_dir()][0]


def load_har(source: Path, part: str):
  signals = np.stack([np.loadtxt(source / part / 'Inertial Signals' / f'{c}_{part}.txt')
                      for c in HAR_CHANNELS], axis=1)
  labels = np.loadtxt(source / part / f'y_{part}.txt').astype(int) - 1
  return signals.astype(np.float32), labels


def prepare_har(root: Path, force: bool = False):
  if is_converted(root) and not force:
    return
  source = fetch_har(root / 'download')
  train, train_y = load_har(source, 'train')
  test, test_y = load_har(source, 'test')

  low = np.percentile(train, 0.5, axis=(0, 2), keepdims=True)
  high = np.percentile(train, 99.5, axis=(0, 2), keepdims=True)
  def scale(x):
    return np.clip(np.round((x - low) / (high - low) * 255.0), 0, 255).astype(np.uint8)

  raw_dir = root / 'MNIST' / 'raw'
  raw_dir.mkdir(parents=True, exist_ok=True)
  for prefix, x, y in (('train', train, train_y), ('t10k', test, test_y)):
    write_idx(scale(x), y, raw_dir, prefix)
    print(f'har {prefix}: {len(y)} windows, {x.shape[1]}x{x.shape[2]}, per class {np.bincount(y).tolist()}')
  print('har; wrote', raw_dir)


PREPARE = {
  'mnist': prepare_mnist,
  'kmnist': prepare_kmnist,
  'fmnist': prepare_fmnist,
  'cifar10': prepare_cifar10,
  'gtsrb': prepare_gtsrb,
  'dvsgesture': prepare_dvsgesture,
  'speechcommands': prepare_speechcommands,
  'ecg': prepare_ecg,
  'gas': prepare_gas,
  'har': prepare_har,
}


def prepare(name: str, dataset_path: Path, **options):
  assert name in OPEN_DATASETS, name
  root = Path(dataset_path).resolve() / name
  root.mkdir(parents=True, exist_ok=True)
  PREPARE[name](root, **options)
  return root


def main():
  parser = argparse.ArgumentParser(description='Download and prepare the open datasets of npx_trainer')
  parser.add_argument('-dataset', '-d', required=True, help='dataset directory')
  parser.add_argument('-i', nargs='+', required=True, choices=OPEN_DATASETS + ('all',),
                      help='datasets to prepare, or all')
  parser.add_argument('-force', action='store_true', help='convert ecg/gas/har again even if converted')
  parser.add_argument('-ecg_protocol', choices=('intra', 'inter'), default='intra')
  parser.add_argument('-gas_protocol', choices=('random', 'drift'), default='random')
  args = parser.parse_args()

  name_list = list(OPEN_DATASETS) if 'all' in args.i else args.i
  for name in name_list:
    print(f'[{name}]')
    options = {}
    if name == 'ecg':
      options = dict(force=args.force, protocol=args.ecg_protocol)
    elif name == 'gas':
      options = dict(force=args.force, protocol=args.gas_protocol)
    elif name == 'har':
      options = dict(force=args.force)
    print('ready:', prepare(name, args.dataset, **options))


if __name__ == '__main__':
  main()
