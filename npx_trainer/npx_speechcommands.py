import os
import math
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Dict

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets import SPEECHCOMMANDS

TASK_3_KEYWORDS = [
    "happy",
]

TASK_6_KEYWORDS = [
    "yes",
    "down",
    "left",
    "right",
    "off",
    "stop",
]

TASK_12_KEYWORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

GSC_35_WORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight",
    "five", "follow", "forward", "four", "go", "happy", "house",
    "learn", "left", "marvin", "nine", "no", "off", "on", "one",
    "right", "seven", "sheila", "six", "stop", "three", "tree",
    "two", "up", "visual", "wow", "yes", "zero",
]

# output_classes: (target keywords, include unknown, include silence)
CLASS_CONFIGS = {
    3: (TASK_3_KEYWORDS, True, True),
    6: (TASK_6_KEYWORDS, False, False),
    12: (TASK_12_KEYWORDS, True, True),
    35: (GSC_35_WORDS, False, False),
}

SUPPORTED_OUTPUT_CLASSES = tuple(CLASS_CONFIGS.keys())

class NpxSpeechCommandsPreprocess(SPEECHCOMMANDS):
    """
    Multi-class Keyword Spotting dataset on Google Speech Commands.

    Classes:
      - one class per target word (in target_words, order preserved)
      - optional 'unknown' and 'silence' classes
      - 3/12 classes include unknown and silence
      - 6/35 classes consist only of target keywords

    Example (target_words=["happy","yes","no"]):
      CLASS_MAP = {"happy":0, "yes":1, "no":2, "unknown":3, "silence":4}

    Features:
      - Multiple target words (each its own class)
      - Diverse unknown sampling (stratified by non-target labels)
      - Imbalance handling:
          * balance_mode="downsample": downsample unknown/silence by ratio per target sample
          * balance_mode="weights": keep all unknown, compute class weights
      - Silence from '_background_noise_' (random crop), zero-silence fallback
      - Optional caching of preprocessed (feature, label) pairs
      - Optional transform: waveform -> feature (e.g., MelSpectrogram + AmplitudeToDB)

    Parameters:
      root: dataset root
      subset: "training" | "validation" | "testing"
      output_classes: one of 3, 6, 12, 35; selects a predefined class configuration
      transform: callable (waveform -> feature). Accepts (n,) or (1,n). Must return (T,M) or (1,T,M).
      target_sr: target sample rate (Hz)
      download: download dataset if missing
      cache_dir: directory path to cache preprocessed tensors (optional)
      verbose: print progress info
      unknown_per_target: unknown count relative to the average size of one target class
      silence_per_target: silence count relative to the average size of one target class
      balance_mode: "downsample" | "weights"
      seed: RNG seed for reproducibility
    """

    def __init__(
        self,
        root: str,
        subset: str,
        output_classes: int = 3,
        transform=None,
        target_sr: int = 16000,
        download: bool = True,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        unknown_per_target: float = 1.0,
        silence_per_target: float = 1.0,
        balance_mode: str = "downsample",
        seed: int = 1337,
    ):
        assert output_classes in CLASS_CONFIGS, (
            f"Unsupported output_classes={output_classes}. "
            f"Expected one of {SUPPORTED_OUTPUT_CLASSES}."
        )
        assert balance_mode in ("downsample", "weights")

        target_words, include_unknown, include_silence = CLASS_CONFIGS[output_classes]

        super().__init__(root, download=download, subset=subset)

        # Basic config
        self.transform = transform
        self.target_sr = target_sr
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.output_classes = output_classes
        self.target_words = [w.lower() for w in target_words]
        self.include_unknown = include_unknown
        self.include_silence = include_silence
        self.unknown_per_target = unknown_per_target
        self.silence_per_target = silence_per_target
        self.balance_mode = balance_mode
        self.rng = random.Random(seed)

        # Parent length and lazy resampler
        self._raw_len = super(NpxSpeechCommandsPreprocess, self).__len__()
        self._resampler = None

        # Create cache directory if needed
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # --- Step 1: Build label -> indices mapping (metadata only) ---
        label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i in range(self._raw_len):
            _, _, label, *_ = super(NpxSpeechCommandsPreprocess, self).__getitem__(i)
            label_to_indices[label].append(i)

        # Warn if some target words are missing
        available_lower = {k.lower() for k in label_to_indices.keys()}
        missing = [w for w in self.target_words if w not in available_lower]
        if missing and self.verbose:
            print(f"[WARN] target words not found in dataset: {missing}")

        # --- Step 2: Build CLASS_MAP dynamically ---
        self.CLASS_MAP: Dict[str, int] = {w: i for i, w in enumerate(self.target_words)}
        if self.include_unknown:
            self.CLASS_MAP["unknown"] = len(self.CLASS_MAP)
        if self.include_silence:
            self.CLASS_MAP["silence"] = len(self.CLASS_MAP)

        assert len(self.CLASS_MAP) == self.output_classes, (
            f"Class configuration mismatch: requested={self.output_classes}, "
            f"built={len(self.CLASS_MAP)}, labels={list(self.CLASS_MAP)}"
        )

        # --- Step 3: Partition indices into targets (per word) and unknown labels ---
        target_indices_by_word: Dict[str, List[int]] = {w: [] for w in self.target_words}
        unknown_label_list: List[str] = []
        for lbl, indices in label_to_indices.items():
            lcl = lbl.lower()
            if lcl in self.target_words:
                target_indices_by_word[lcl].extend(indices)
            elif self.include_unknown and "_background_" not in lcl:
                unknown_label_list.append(lbl)

        # Gather all targets together for counting
        target_all_indices: List[int] = []
        for w in self.target_words:
            target_all_indices.extend(target_indices_by_word[w])

        if len(target_all_indices) == 0:
            raise ValueError("No target samples found. Check target_words or dataset root/subset.")

        # Unknown candidates pooled per non-target label
        unknown_pool_by_label = {lbl: label_to_indices[lbl][:] for lbl in unknown_label_list}

        # --- Step 4: Decide desired counts for unknown and silence ---
        # Use the average number of samples in one target keyword class.
        # Therefore, a ratio of 1.0 makes unknown/silence approximately the
        # same size as one average target class, regardless of whether the
        # configuration contains 1 target keyword (3 classes) or 10 (12 classes).
        n_target = len(target_all_indices)
        num_target_classes = len(self.target_words)
        assert num_target_classes > 0, "At least one target word is required."

        samples_per_target_class = n_target / num_target_classes

        if self.include_unknown:
            if self.balance_mode == "downsample":
                n_unknown_desired = int(round(
                    self.unknown_per_target * samples_per_target_class
                ))
            else:
                # Keep all unknown samples in weights mode.
                n_unknown_desired = sum(
                    len(indices) for indices in unknown_pool_by_label.values()
                )
        else:
            n_unknown_desired = 0

        if self.include_silence:
            n_silence_desired = int(round(
                self.silence_per_target * samples_per_target_class
            ))
        else:
            n_silence_desired = 0

        # --- Step 5: Stratified sampling for diverse unknowns ---
        unknown_indices = self._sample_unknown_diverse(
            unknown_pool_by_label,
            desired=n_unknown_desired,
            rng=self.rng,
        )

        # --- Step 6: Load background noise for silence creation ---
        bg_noises = self._load_background_noise_files()

        # --- Step 7: Build final item spec list ---
        # Each item is a tuple (type, src):
        #   type ¡ô {"target", "unknown", "silence"}
        #   src  ¡ô int (dataset index) for target/unknown, or None for silence
        item_specs: List[Tuple[str, object]] = []

        # Add ALL target indices (across all target words)
        for idx in target_all_indices:
            item_specs.append(("target", idx))

        # Add unknown (sampled or all, depending on mode)
        for idx in unknown_indices:
            item_specs.append(("unknown", idx))

        # Add silence specs
        for _ in range(max(0, n_silence_desired)):
            item_specs.append(("silence", None))

        # Shuffle dataset order
        self.rng.shuffle(item_specs)

        if self.verbose:
            print(
                f"[{subset}] Build: target={len(target_all_indices)} "
                f"(split across {len(self.target_words)} words), "
                f"unknown={len(unknown_indices)}, silence={n_silence_desired} "
                f"(mode={self.balance_mode}, num_classes={self.num_classes()})"
            )

        # --- Step 8: Preprocess -> (feature, label), cache if requested ---
        self.data: List[torch.Tensor] = []
        self.targets: List[int] = []
        self.class_weights: Optional[torch.Tensor] = None
        self._bg_noises_cache = bg_noises

        for i, (typ, src) in enumerate(item_specs):
            # Cache fast-path
            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, f"{i:07d}.pt")
                if os.path.exists(cache_path):
                    obj = torch.load(cache_path)
                    self.data.append(obj["feature"])
                    self.targets.append(int(obj["label"]))
                    continue

            if typ in ("target", "unknown"):
                waveform, sr, label, *_ = super(NpxSpeechCommandsPreprocess, self).__getitem__(src)
                assert sr == self.target_sr
                feature = self._prepare_feature_from_waveform(waveform)
                lcl = label.lower()
                if lcl in self.target_words:
                    # Each target word has its own class index
                    label = self.CLASS_MAP[lcl]
                elif self.include_unknown:
                    label = self.CLASS_MAP["unknown"]
                else:
                    raise RuntimeError(
                        f"Non-target label '{label}' appeared without an unknown class."
                    )
            elif typ == "silence":
                assert self.include_silence
                num_samples = 32000
                waveform = self._sample_silence_waveform(self._bg_noises_cache, num_samples)
                feature = self._prepare_feature_from_waveform(waveform)
                label = self.CLASS_MAP["silence"]
            else:
                raise RuntimeError("Invalid item type")

            self.data.append(feature)
            self.targets.append(int(label))

            if self.cache_dir is not None:
                torch.save({"feature": feature, "label": int(label)}, cache_path)

        # Try stacking (only if all shapes are consistent)
        try:
            self.data = torch.stack(self.data)  # (N, C, T, M) or (N, 1, L)
        except Exception:
            if self.verbose:
                print("Data has varying shapes; keeping as list.")

        self.targets = torch.tensor(self.targets, dtype=torch.long)

        # --- Step 9: Optional class weights (useful for CrossEntropyLoss) ---
        if self.balance_mode == "weights":
            counts = torch.bincount(self.targets, minlength=self.num_classes()).float()
            weights = counts.sum() / counts.clamp_min(1.0)
            self.class_weights = weights
            if self.verbose:
                print(f"[{subset}] class_counts={counts.tolist()}, class_weights={weights.tolist()}")

        if self.verbose:
            shape_info = self.data.shape if isinstance(self.data, torch.Tensor) else f"list(len={len(self.data)})"
            print(f"[{subset}] Done. data={shape_info}, targets={self.targets.shape}")

    # ---------------------------
    # Public API
    # ---------------------------
    def __getitem__(self, index):
        if self.cache_dir is not None:
            obj = torch.load(os.path.join(self.cache_dir, f"{index:07d}.pt"))
            return obj["feature"], obj["label"]

        if isinstance(self.data, torch.Tensor):
            return self.data[index], int(self.targets[index])
        else:
            return self.data[index], int(self.targets[index])

    def __len__(self):
        if hasattr(self, "targets") and len(self.targets) > 0:
            return len(self.targets)
        if hasattr(self, "_raw_len"):
            return self._raw_len
        return super(NpxSpeechCommandsPreprocess, self).__len__()

    def num_classes(self) -> int:
        """Return the configured number of output classes."""
        return len(self.CLASS_MAP)

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Return per-class weights if balance_mode='weights', else None."""
        return self.class_weights

    def idx_to_label(self, idx: int) -> str:
        """Return human-readable class name by index."""
        for k, v in self.CLASS_MAP.items():
            if v == idx:
                return k
        return f"<unknown-{idx}>"

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _prepare_feature_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Resample -> Pad/Trim -> Transform.
        Output shape:
          - If transform returns (T, M), expand to (1, T, M)
          - Otherwise keep raw waveform as (1, 1, L)
        """
        if self.transform is not None:
            feat = self.transform(waveform.squeeze(0))
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)  # (1, T, M)
        else:
            feat = waveform.unsqueeze(0)  # (1, 1, L)

        return feat

    def _sample_unknown_diverse(self, pool_by_label: Dict[str, List[int]], desired: int, rng: random.Random) -> List[int]:
        """
        Stratified sampling across non-target labels to ensure diversity in 'unknown'.
        """
        labels = list(pool_by_label.keys())
        if desired <= 0 or len(labels) == 0:
            return []

        total_avail = sum(len(v) for v in pool_by_label.values())
        desired = min(desired, total_avail)

        # Shuffle each label pool in-place
        for lbl in labels:
            rng.shuffle(pool_by_label[lbl])

        # First pass: equal quota per label
        base_quota = max(1, desired // max(1, len(labels)))
        out = []
        remainder = desired

        for lbl in labels:
            take = min(base_quota, len(pool_by_label[lbl]))
            out.extend(pool_by_label[lbl][:take])
            pool_by_label[lbl] = pool_by_label[lbl][take:]
            remainder -= take

        # Second pass: fill remainder by cycling labels with leftovers
        lbl_idx = 0
        while remainder > 0:
            if all(len(pool_by_label[lbl]) == 0 for lbl in labels):
                break
            lbl = labels[lbl_idx % len(labels)]
            if len(pool_by_label[lbl]) > 0:
                out.append(pool_by_label[lbl].pop(0))
                remainder -= 1
            lbl_idx += 1

        rng.shuffle(out)
        return out

    def _load_background_noise_files(self) -> List[torch.Tensor]:
        """
        Load waveforms from '_background_noise_' (mono), resample if needed.
        Returns a list of (1, N) tensors. If none, returns [].
        """
        noises: List[torch.Tensor] = []
        base = getattr(self, "_path", None) or self._path  # torchaudio's internal root
        bg_dir = os.path.join(base, "_background_noise_")
        if os.path.isdir(bg_dir):
            for fn in os.listdir(bg_dir):
                if not fn.lower().endswith((".wav", ".flac")):
                    continue
                p = os.path.join(bg_dir, fn)
                wav, sr = torchaudio.load(p)  # (C, N)
                assert sr == self.target_sr
                if wav.dim() == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)  # convert to mono
                elif wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                noises.append(wav)

        if self.verbose:
            print(
                f"Background noise files: {len(noises)} found."
                if noises else "No background noise found. Using zero-silence."
            )
        return noises

    def _sample_silence_waveform(self, bg_noises: List[torch.Tensor], num_samples: int) -> torch.Tensor:
        """
        Create a 'silence' segment by randomly cropping a background noise file.
        If none available, return zero-silence of length num_samples.
        """
        if bg_noises:
            wav = bg_noises[self.rng.randrange(len(bg_noises))]
            N = wav.size(1)
            if N >= num_samples:
                start = self.rng.randrange(0, N - num_samples + 1)
                seg = wav[:, start:start + num_samples]
            else:
                reps = (num_samples + N - 1) // N
                seg = wav.repeat(1, reps)[:, :num_samples]
            return seg
        else:
            return torch.zeros(1, num_samples)

class NpxMelSpectrogram(torch.nn.Module):
    __constants__ = ["num_samples", "sample_rate", "n_fft", "win_length", "hop_length", "n_mels"]

    def __init__(
        self,
        num_samples: int = 16000,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        pre_emp: bool = True,
        window: bool = True,
    ) -> None:
        super(NpxMelSpectrogram, self).__init__()

        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.n_mels = n_mels
        self.pre_emp = pre_emp
        self.window = window
        self.fbank = self.get_filter_bank()

    @staticmethod
    def _pad_or_trim(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Pad with zeros or trim to a fixed number of samples."""
        n = waveform.size(1)
        if n < num_samples:
            pad = num_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :num_samples]
        return waveform

    def preemphasis(self, waveform: Tensor, pre_emphasis=0.97) -> Tensor:
        # pre-emphasis per sample in batch
        emphasized = torch.cat(
            (waveform[:, 0:1], waveform[:, 1:] - pre_emphasis * waveform[:, :-1]), dim=1
        )
        return emphasized

    def framing(self, waveform: Tensor) -> Tensor:
        """
        waveform: (batch, waveform_length)
        return: (batch, num_frames, win_length)
        """
        batch_size, waveform_length = waveform.shape
        assert waveform_length >= self.win_length, "must be waveform_length >= window length"
        num_frames = int(math.floor(float(waveform_length - self.win_length) / self.hop_length)) + 1 # without pad
        #num_frames = int(math.ceil(float(waveform_length - self.win_length) / self.hop_length)) + 1 # with pad
        #pad_waveform_length = (num_frames - 1) * self.hop_length + self.win_length
        #pad_len = pad_waveform_length - waveform_length
        #if pad_len > 0:
        #    pad = torch.zeros(batch_size, pad_len, dtype=waveform.dtype, device=waveform.device)
        #    waveform = torch.cat((waveform, pad), dim=1)

        # frame indices
        indices0 = torch.arange(0, self.win_length, device=waveform.device).unsqueeze(0).repeat(num_frames, 1)
        indices1 = torch.arange(0, num_frames * self.hop_length, self.hop_length, device=waveform.device).unsqueeze(1)
        indices = indices0 + indices1  # (num_frames, win_length)

        frames = waveform.unfold(dimension=1, size=self.win_length, step=self.hop_length)
        return frames  # (batch, num_frames, win_length)

    def windowing(self, frames: Tensor) -> Tensor:
        hamming = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(self.win_length, device=frames.device) / (self.win_length - 1)
        )
        return frames * hamming  # broadcast along batch & frames

    def fourer_transform(self, frames: Tensor) -> Tensor:
        # frames: (batch, num_frames, win_length)
        # -> (batch, num_frames, n_fft//2 + 1)
        dft = torch.fft.rfft(frames, n=self.n_fft)
        return dft

    def get_filter_bank(self) -> Tensor:
        low_freq_mel = 0.0
        high_freq_mel = 2595 * torch.log10(torch.tensor(1.0 + (self.sample_rate / 2) / 700.0))
        mel_points = torch.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()

        fbank = torch.zeros((self.n_mels, self.n_fft // 2 + 1), dtype=torch.float64)
        for m in range(1, self.n_mels + 1):
            f_m_minus, f_m, f_m_plus = bin[m - 1], bin[m], bin[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fbank

    def filter_bank(self, spectrum: Tensor) -> Tensor:
        # spectrum: (batch, num_frames, n_fft//2+1)
        fbank = self.fbank.to(spectrum.device).to(spectrum.dtype)
        out = torch.matmul(spectrum, fbank.T)
        eps = torch.finfo(out.dtype).eps
        out = torch.where(out == 0, torch.tensor(eps, device=out.device, dtype=out.dtype), out)
        return 10 * torch.log10(out)

    def forward(self, waveform: Tensor) -> Tensor:
        """
        waveform shape:
          - (n,) : single waveform
          - (1, n): single batch waveform
          - (B, n): batch input
        return: (B, num_frames, n_mels)
        """
        debug = False
        # unify input shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.size(0) == 1:
            pass
        elif waveform.dim() != 2:
            raise ValueError(f"Unsupported input shape: {waveform.shape}")

        waveform = self._pad_or_trim(waveform, self.num_samples)

        if debug: print('waveform', waveform.shape, waveform)
        if self.pre_emp:
            waveform = self.preemphasis(waveform)
        if debug: print('pre_emp', waveform.shape, waveform)

        frames = self.framing(waveform)

        if debug: print('frames', frames.shape, frames)
        if self.window:
            frames = self.windowing(frames)
        if debug: print('windowing frames', frames.shape, frames)

        dft = self.fourer_transform(frames)
        if debug: print('dft', dft.shape, dft)
        mag = torch.abs(dft)
        if debug: print('mag', mag.shape, mag)
        pow_spec = (1.0 / self.n_fft) * (mag ** 2)
        if debug: print('pow_spec', pow_spec.shape, pow_spec)
        mel = self.filter_bank(pow_spec)
        return mel  # (B, num_frames, n_mels)

