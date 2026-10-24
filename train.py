"""Train the photo classifier and the recording classifier.

Weights are updated only on the train split. The val split chooses the
checkpoint. The calib and test splits are not read.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from download_dataset import SPECIES

ROOT = Path(__file__).resolve().parent
SPLITS_PATH = ROOT / "data" / "metadata" / "splits.csv"
LOG_PATH = ROOT / "data" / "metadata" / "training_log.csv"
SKIP_PATH = ROOT / "data" / "metadata" / "audio_decode_skips.csv"
MODEL_DIR = ROOT / "models"
MEL_CACHE = ROOT / "data" / "cache" / "mel"

SAMPLE_RATE = 22050
CLIP_SECONDS = 3.0
MAX_AUDIO_WINDOWS = 8
MAX_AUDIO_SECONDS = 120.0
N_MELS = 64
MEL_FRAMES = 128
IMAGE_EPOCHS = 6
AUDIO_EPOCHS = 8
PATIENCE = 2
BATCH_SIZE = 32
SEED = 42

LABELS = [common for common, _ in SPECIES]
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABELS)}

LOG_FIELDS = ["modality", "epoch", "train_loss", "train_acc", "val_acc", "seconds", "kept_checkpoint"]


def read_split(modality: str, split_name: str) -> list[dict[str, str]]:
    with SPLITS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["modality"] == modality and row["split"] == split_name
        ]
    if not rows:
        raise RuntimeError(f"no {split_name} rows for {modality}")
    unknown = sorted({row["common_name"] for row in rows} - set(LABEL_TO_INDEX))
    if unknown:
        raise RuntimeError(f"unlabeled species: {unknown}")
    return rows


class ImageSet(Dataset):
    def __init__(self, rows: list[dict[str, str]], train: bool) -> None:
        self.rows = rows
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(ROOT / row["file_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABEL_TO_INDEX[row["common_name"]]


def decode_with_av(path: Path, need: int, pad: bool = True) -> np.ndarray:
    container = av.open(str(path))
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    got = 0
    try:
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            if converted is None:
                continue
            if not isinstance(converted, list):
                converted = [converted]
            for item in converted:
                samples = np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1)
                chunks.append(samples)
                got += int(samples.shape[0])
            if got >= need:
                break
        tail = resampler.resample(None)
        if tail:
            if not isinstance(tail, list):
                tail = [tail]
            for item in tail:
                samples = np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1)
                chunks.append(samples)
    finally:
        container.close()
    if not chunks:
        raise RuntimeError(f"no audio frames in {path}")
    audio = np.concatenate(chunks).astype(np.float32)
    if not pad:
        return audio[:need]
    return fit_length(audio, need)


def decode_clip(path: Path) -> np.ndarray:
    """Return CLIP_SECONDS of mono audio at SAMPLE_RATE."""
    need = int(SAMPLE_RATE * CLIP_SECONDS)
    errors: list[str] = []
    if path.suffix.lower() == ".wav":
        try:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if file_sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SAMPLE_RATE)
            return fit_length(audio.astype(np.float32), need)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")
    try:
        return decode_with_av(path, need)
    except Exception as exc:
        errors.append(f"av: {exc}")
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
        return fit_length(np.asarray(audio, dtype=np.float32), need)
    except Exception as exc:
        errors.append(f"librosa: {exc}")
    raise RuntimeError("; ".join(errors))


def decode_capped(path: Path, max_samples: int) -> np.ndarray:
    """Return up to max_samples of mono audio. Short files are not padded."""
    errors: list[str] = []
    if path.suffix.lower() == ".wav":
        try:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if file_sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SAMPLE_RATE)
            return np.asarray(audio[:max_samples], dtype=np.float32)
        except Exception as exc:
            errors.append(f"soundfile: {exc}")
    try:
        return decode_with_av(path, max_samples, pad=False)
    except Exception as exc:
        errors.append(f"av: {exc}")
    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=max_samples / SAMPLE_RATE)
        return np.asarray(audio[:max_samples], dtype=np.float32)
    except Exception as exc:
        errors.append(f"librosa: {exc}")
    raise RuntimeError("; ".join(errors))


def extra_window_starts(n_samples: int, window: int, max_windows: int) -> list[int]:
    """Starts after the opening window. At most max_windows - 1, spread across the file."""
    if n_samples <= window or max_windows < 2:
        return []
    extras = min(max_windows - 1, int(np.ceil(n_samples / window)) - 1)
    if extras <= 0:
        return []
    last = n_samples - window
    if extras == 1:
        return [int(last)] if last >= window // 2 else []
    starts: list[int] = []
    for index in range(1, extras + 1):
        start = int(round(index * last / extras))
        if start < window // 2:
            continue
        if not starts or start != starts[-1]:
            starts.append(start)
    return starts


def fit_length(audio: np.ndarray, need: int) -> np.ndarray:
    if audio.shape[0] < need:
        return np.pad(audio, (0, need - audio.shape[0]))
    return audio[:need]


def log_mel(audio: np.ndarray) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS,
        center=True,
    )
    peak = float(np.max(spectrogram)) if spectrogram.size else 0.0
    decibels = librosa.power_to_db(spectrogram, ref=peak if peak > 0 else 1.0).astype(np.float32)
    if decibels.shape[1] < MEL_FRAMES:
        decibels = np.pad(decibels, ((0, 0), (0, MEL_FRAMES - decibels.shape[1])))
    return decibels[:, :MEL_FRAMES]


def mel_image_tensor(mel: np.ndarray) -> torch.Tensor:
    """Turn one log-mel into the 3-channel image the recording MobileNet scores."""
    tensor = torch.from_numpy(np.ascontiguousarray(mel)).unsqueeze(0).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0).repeat(3, 1, 1)
    low = torch.quantile(tensor, 0.01)
    high = torch.quantile(tensor, 0.99)
    tensor = (tensor - low) / (high - low + 1e-6)
    tensor = tensor.clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def cache_mel(row: dict[str, str]) -> Path:
    MEL_CACHE.mkdir(parents=True, exist_ok=True)
    cache = MEL_CACHE / f"{row['sample_id']}.npy"
    if not cache.is_file():
        np.save(cache, log_mel(decode_clip(ROOT / row["file_path"])))
    return cache


def usable_audio_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Cache each clip and drop recordings that no decoder can read."""
    kept: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            cache_mel(row)
            kept.append(row)
        except Exception as exc:
            skipped.append(
                {
                    "sample_id": row["sample_id"],
                    "file_path": row["file_path"],
                    "split": row["split"],
                    "common_name": row["common_name"],
                    "error": str(exc).replace("\n", " ")[:300],
                }
            )
            print(f"skip audio {row['file_path']}: {exc}", flush=True)
        if index % 100 == 0:
            print(f"audio cache {index}/{len(rows)} kept {len(kept)} skipped {len(skipped)}", flush=True)
    if skipped:
        new_file = not SKIP_PATH.exists()
        with SKIP_PATH.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "file_path", "split", "common_name", "error"],
                lineterminator="\n",
            )
            if new_file:
                writer.writeheader()
            writer.writerows(skipped)
    return kept


class AudioSet(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        MEL_CACHE.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        mel = np.load(cache_mel(row))
        tensor = torch.from_numpy(np.ascontiguousarray(mel)).unsqueeze(0)
        return tensor, LABEL_TO_INDEX[row["common_name"]]


class MelNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.features(inputs).flatten(1)
        return self.classifier(hidden)


def image_model() -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(LABELS))
    return model


def image_model_finetune(checkpoint_path: Path) -> tuple[nn.Module, float]:
    """Continue from the saved head and unfreeze the last feature blocks."""
    model = image_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return model, float(checkpoint.get("val_acc", 0.0))


def audio_mobilenet() -> nn.Module:
    """ImageNet MobileNet on a log-mel image. The small MelNet was too weak."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(LABELS))
    return model


class MelImageSet(Dataset):
    def __init__(self, rows: list[dict[str, str]], train: bool) -> None:
        self.rows = rows
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        tensor = mel_image_tensor(np.load(cache_mel(row)))
        if self.train and torch.rand(1).item() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        return tensor, LABEL_TO_INDEX[row["common_name"]]


def audio_window_logits(model: nn.Module, row: dict[str, str]) -> tuple[np.ndarray, int]:
    """Average the saved recording model over 3-second windows.

    The first window is the same clip the model was trained on. Later windows
    are spread across the rest of the file, up to MAX_AUDIO_WINDOWS in total
    and MAX_AUDIO_SECONDS of audio.
    """
    first, _label = MelImageSet([row], train=False)[0]
    window = int(SAMPLE_RATE * CLIP_SECONDS)
    tensors = [first]
    try:
        audio = decode_capped(ROOT / row["file_path"], int(SAMPLE_RATE * MAX_AUDIO_SECONDS))
        for start in extra_window_starts(int(audio.shape[0]), window, MAX_AUDIO_WINDOWS):
            chunk = fit_length(audio[start : start + window], window)
            tensors.append(mel_image_tensor(log_mel(chunk)))
    except Exception as exc:
        print(f"extra windows skipped {row['file_path']}: {exc}", flush=True)
        tensors = [first]
    batch = torch.stack(tensors)
    with torch.no_grad():
        averaged = model(batch).mean(dim=0)
    return averaged.numpy().astype(np.float32), len(tensors)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, targets in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * targets.shape[0]
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        seen += int(targets.shape[0])
    return total_loss / seen, correct / seen


def append_log(row: dict[str, str]) -> None:
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def fit_model(
    modality: str,
    model: nn.Module,
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    epochs: int,
    destination: Path,
    floor_accuracy: float,
    backbone_lr: float,
    patience: int = PATIENCE,
) -> None:
    if modality == "image_ft":
        train_data = ImageSet(train_rows, train=True)
        val_data = ImageSet(val_rows, train=False)
    else:
        train_data = MelImageSet(train_rows, train=True)
        val_data = MelImageSet(val_rows, train=False)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    backbone = [parameter for parameter in model.features.parameters() if parameter.requires_grad]
    head = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": 1e-3},
        ]
    )
    best_accuracy = floor_accuracy
    stale = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainable = sum(parameter.requires_grad for parameter in model.parameters())
    print(
        f"{modality}: train {len(train_rows)} val {len(val_rows)} "
        f"epochs {epochs} trainable_tensors {trainable} floor {floor_accuracy:.3f}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer)
        _, val_accuracy = run_epoch(model, val_loader, None)
        kept = "no"
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale = 0
            kept = "yes"
            torch.save(
                {
                    "modality": modality,
                    "labels": LABELS,
                    "state_dict": model.state_dict(),
                    "val_acc": val_accuracy,
                    "epoch": epoch,
                },
                destination,
            )
        else:
            stale += 1
        elapsed = time.perf_counter() - started
        print(
            f"{modality} epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f} "
            f"best={best_accuracy:.3f}",
            flush=True,
        )
        append_log(
            {
                "modality": modality,
                "epoch": str(epoch),
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_accuracy:.6f}",
                "val_acc": f"{val_accuracy:.6f}",
                "seconds": f"{elapsed:.1f}",
                "kept_checkpoint": kept,
            }
        )
        if stale >= patience:
            print(f"{modality} stopped after {patience} epochs without a better val accuracy", flush=True)
            break
    print(f"{modality} best val_acc={best_accuracy:.3f} saved {destination}", flush=True)


def improve() -> None:
    """Fine-tune the photo network and replace the weak recording network."""
    image_checkpoint = MODEL_DIR / "image_best.pt"
    audio_checkpoint = MODEL_DIR / "audio_best.pt"
    if not image_checkpoint.is_file():
        raise RuntimeError("missing models/image_best.pt")
    head_copy = MODEL_DIR / "image_head_best.pt"
    if not head_copy.is_file():
        head_copy.write_bytes(image_checkpoint.read_bytes())
    if audio_checkpoint.is_file():
        melnet_copy = MODEL_DIR / "audio_melnet_best.pt"
        if not melnet_copy.is_file():
            melnet_copy.write_bytes(audio_checkpoint.read_bytes())

    image_rows_train = read_split("image", "train")
    image_rows_val = read_split("image", "val")
    model, floor = image_model_finetune(head_copy)
    fit_model(
        "image_ft",
        model,
        image_rows_train,
        image_rows_val,
        epochs=8,
        destination=image_checkpoint,
        floor_accuracy=floor,
        backbone_lr=1e-4,
        patience=3,
    )

    if SKIP_PATH.exists():
        SKIP_PATH.unlink()
    audio_rows_train = usable_audio_rows(read_split("audio", "train"))
    audio_rows_val = usable_audio_rows(read_split("audio", "val"))
    print(f"audio usable train {len(audio_rows_train)} val {len(audio_rows_val)}", flush=True)
    fit_model(
        "audio_ft",
        audio_mobilenet(),
        audio_rows_train,
        audio_rows_val,
        epochs=10,
        destination=audio_checkpoint,
        floor_accuracy=0.0,
        backbone_lr=1e-4,
        patience=3,
    )


def train_modality(modality: str) -> None:
    train_rows = read_split(modality, "train")
    val_rows = read_split(modality, "val")
    print(
        f"{modality}: train {len(train_rows)} val {len(val_rows)} "
        f"classes {len(LABELS)} device cpu",
        flush=True,
    )
    if modality == "image":
        train_data = ImageSet(train_rows, train=True)
        val_data = ImageSet(val_rows, train=False)
        model = image_model()
        epochs = IMAGE_EPOCHS
    else:
        if SKIP_PATH.exists():
            SKIP_PATH.unlink()
        train_rows = usable_audio_rows(train_rows)
        val_rows = usable_audio_rows(val_rows)
        print(f"audio usable train {len(train_rows)} val {len(val_rows)}", flush=True)
        train_data = AudioSet(train_rows)
        val_data = AudioSet(val_rows)
        model = MelNet(len(LABELS))
        epochs = AUDIO_EPOCHS

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    best_accuracy = -1.0
    stale = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / f"{modality}_best.pt"

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, None)
        kept = "no"
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale = 0
            kept = "yes"
            torch.save(
                {
                    "modality": modality,
                    "labels": LABELS,
                    "state_dict": model.state_dict(),
                    "val_acc": val_accuracy,
                    "epoch": epoch,
                },
                destination,
            )
        else:
            stale += 1
        elapsed = time.perf_counter() - started
        print(
            f"{modality} epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f} "
            f"best={best_accuracy:.3f}",
            flush=True,
        )
        append_log(
            {
                "modality": modality,
                "epoch": str(epoch),
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_accuracy:.6f}",
                "val_acc": f"{val_accuracy:.6f}",
                "seconds": f"{elapsed:.1f}",
                "kept_checkpoint": kept,
            }
        )
        if stale >= PATIENCE:
            print(f"{modality} stopped after {PATIENCE} epochs without a better val accuracy", flush=True)
            break
    print(f"{modality} best val_acc={best_accuracy:.3f} saved {destination}", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    requested = sys.argv[1:] or ["image", "audio"]
    if requested == ["improve"]:
        print("fine-tuning photos and recordings on the train split; calib and test stay unread", flush=True)
        improve()
        print("TRAINING_DONE", flush=True)
        return
    if requested == ["image", "audio"] and LOG_PATH.exists():
        LOG_PATH.unlink()
    print("training on the train split only; val selects the checkpoint; calib and test stay unread", flush=True)
    for modality in requested:
        train_modality(modality)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
