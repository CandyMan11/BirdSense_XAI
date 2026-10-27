"""Fuse the photo and recording models.

Photos and recordings come from different observations, so a training pair is
one photo and one recording of the same species from the same split.
The fusion rule is chosen on val. Temperature is fit on calib.
The test split is scored once, after that choice is fixed.
`python fuse.py windows` may score it one more time, and only if
whole-file recording windows beat the saved validation accuracy.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train
from train import (
    LABELS,
    LABEL_TO_INDEX,
    ROOT,
    ImageSet,
    MelImageSet,
    audio_mobilenet,
    cache_mel,
    image_model,
    read_split,
)

LOGIT_DIR = ROOT / "data" / "cache" / "logits"
RESULT_PATH = ROOT / "data" / "metadata" / "fusion_results.csv"
SPECIES_PATH = ROOT / "data" / "metadata" / "fusion_test_by_species.csv"
PAIRS_PER_IMAGE = 4
FUSION_EPOCHS = 40
SEED = 42
WINDOW_SYSTEM = "linear_windows"
WINDOW_VAL_FLOOR = 0.638


class WeightedSum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(2))

    def forward(self, image_logits: torch.Tensor, audio_logits: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.raw, dim=0)
        return weights[0] * image_logits + weights[1] * audio_logits

    def weights(self) -> tuple[float, float]:
        weights = torch.softmax(self.raw.detach(), dim=0)
        return float(weights[0]), float(weights[1])


class LogitFuse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(len(LABELS) * 2, len(LABELS))

    def forward(self, image_logits: torch.Tensor, audio_logits: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([image_logits, audio_logits], dim=1))


class PairSet(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str, int]],
        image_logits: dict[str, np.ndarray],
        audio_logits: dict[str, np.ndarray],
    ) -> None:
        self.pairs = pairs
        self.image_logits = image_logits
        self.audio_logits = audio_logits

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_id, audio_id, label = self.pairs[index]
        return (
            torch.from_numpy(self.image_logits[image_id]),
            torch.from_numpy(self.audio_logits[audio_id]),
            label,
        )


def load_models() -> tuple[nn.Module, nn.Module]:
    image = image_model()
    image_checkpoint = torch.load(ROOT / "models" / "image_best.pt", map_location="cpu", weights_only=False)
    if image_checkpoint.get("labels") != LABELS:
        raise RuntimeError("image checkpoint labels do not match the species list")
    image.load_state_dict(image_checkpoint["state_dict"])
    image.eval()

    audio = audio_mobilenet()
    audio_checkpoint = torch.load(ROOT / "models" / "audio_best.pt", map_location="cpu", weights_only=False)
    if audio_checkpoint.get("labels") != LABELS:
        raise RuntimeError("audio checkpoint labels do not match the species list")
    audio.load_state_dict(audio_checkpoint["state_dict"])
    audio.eval()
    return image, audio


def usable_audio(split_name: str) -> list[dict[str, str]]:
    rows = read_split("audio", split_name)
    kept = []
    for index, row in enumerate(rows, start=1):
        try:
            cache_mel(row)
            kept.append(row)
        except Exception as exc:
            print(f"skip {split_name} {row['file_path']}: {exc}", flush=True)
        if index % 100 == 0 or index == len(rows):
            print(f"audio prepare {split_name} {index}/{len(rows)} kept {len(kept)}", flush=True)
    return kept


@torch.no_grad()
def embed_split(
    model: nn.Module,
    rows: list[dict[str, str]],
    modality: str,
    split_name: str,
    directory: Path = LOGIT_DIR,
) -> dict[str, np.ndarray]:
    cache_path = directory / f"{modality}_{split_name}.npz"
    if cache_path.is_file():
        stored = np.load(cache_path, allow_pickle=False)
        print(f"loaded {cache_path.name} {len(stored['ids'])}", flush=True)
        return {str(sample_id): logits for sample_id, logits in zip(stored["ids"], stored["logits"])}

    if modality == "image":
        dataset = ImageSet(rows, train=False)
    else:
        dataset = MelImageSet(rows, train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    chunks = []
    seen = 0
    for inputs, _targets in loader:
        chunks.append(model(inputs).numpy())
        seen += inputs.shape[0]
        print(f"logits {modality} {split_name} {seen}/{len(rows)}", flush=True)
    logits = np.concatenate(chunks).astype(np.float32)
    ids = np.array([row["sample_id"] for row in rows])
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ids=ids, logits=logits)
    return {sample_id: row_logits for sample_id, row_logits in zip(ids.tolist(), logits)}


def build_pairs(
    split_name: str,
    image_logits: dict[str, np.ndarray],
    audio_logits: dict[str, np.ndarray],
    image_rows: list[dict[str, str]],
    audio_rows: list[dict[str, str]],
    exhaustive: bool,
) -> list[tuple[str, str, int]]:
    images: dict[str, list[str]] = {name: [] for name in LABELS}
    audios: dict[str, list[str]] = {name: [] for name in LABELS}
    for row in image_rows:
        if row["sample_id"] in image_logits:
            images[row["common_name"]].append(row["sample_id"])
    for row in audio_rows:
        if row["sample_id"] in audio_logits:
            audios[row["common_name"]].append(row["sample_id"])
    rng = np.random.default_rng(SEED)
    pairs: list[tuple[str, str, int]] = []
    for name in LABELS:
        species_images = images[name]
        species_audio = audios[name]
        if not species_images or not species_audio:
            print(f"no pairs for {name} in {split_name}", flush=True)
            continue
        label = LABEL_TO_INDEX[name]
        if exhaustive:
            for image_id in species_images:
                for audio_id in species_audio:
                    pairs.append((image_id, audio_id, label))
        else:
            for image_id in species_images:
                chosen = rng.choice(species_audio, size=PAIRS_PER_IMAGE, replace=True)
                for audio_id in chosen.tolist():
                    pairs.append((image_id, audio_id, label))
    print(f"pairs {split_name} {len(pairs)}", flush=True)
    return pairs


def batch_logits(
    system: str,
    module: nn.Module | None,
    image_batch: torch.Tensor,
    audio_batch: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if system == "mean_prob":
        mixed = 0.5 * torch.softmax(image_batch / temperature, dim=1) + 0.5 * torch.softmax(audio_batch / temperature, dim=1)
        return torch.log(mixed.clamp_min(1e-8))
    if system == "image_only":
        fused = image_batch
    elif system == "audio_only":
        fused = audio_batch
    else:
        if module is None:
            raise RuntimeError(f"missing module for {system}")
        fused = module(image_batch, audio_batch)
    return fused / temperature


def score(
    system: str,
    module: nn.Module | None,
    pairs: list[tuple[str, str, int]],
    image_logits: dict[str, np.ndarray],
    audio_logits: dict[str, np.ndarray],
    temperature: float = 1.0,
) -> tuple[float, float, list[int], list[int]]:
    if not pairs:
        return 0.0, 0.0, [], []
    loader = DataLoader(PairSet(pairs, image_logits, audio_logits), batch_size=256, shuffle=False)
    correct = 0
    total_nll = 0.0
    seen = 0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for image_batch, audio_batch, target in loader:
            logits = batch_logits(system, module, image_batch, audio_batch, temperature)
            loss = F.cross_entropy(logits, target, reduction="sum")
            predicted = logits.argmax(dim=1)
            correct += int((predicted == target).sum().item())
            total_nll += float(loss.item())
            seen += int(target.shape[0])
            labels.extend(target.tolist())
            predictions.extend(predicted.tolist())
    return correct / seen, total_nll / seen, labels, predictions


def train_module(
    name: str,
    module: nn.Module,
    train_pairs: list[tuple[str, str, int]],
    val_pairs: list[tuple[str, str, int]],
    image_by_split: dict[str, dict[str, np.ndarray]],
    audio_by_split: dict[str, dict[str, np.ndarray]],
) -> nn.Module:
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3 if name == "linear" else 1e-2)
    train_loader = DataLoader(
        PairSet(train_pairs, image_by_split["train"], audio_by_split["train"]),
        batch_size=64,
        shuffle=True,
    )
    best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    best_accuracy = -1.0
    stale = 0
    for epoch in range(1, FUSION_EPOCHS + 1):
        module.train()
        for image_batch, audio_batch, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(module(image_batch, audio_batch), target)
            loss.backward()
            optimizer.step()
        module.eval()
        accuracy, nll, _, _ = score(name, module, val_pairs, image_by_split["val"], audio_by_split["val"])
        print(f"{name} epoch {epoch} val_acc={accuracy:.3f} val_nll={nll:.3f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    module.load_state_dict(best_state)
    module.eval()
    print(f"{name} selected val_acc={best_accuracy:.3f}", flush=True)
    return module


def fit_temperature(
    system: str,
    module: nn.Module | None,
    pairs: list[tuple[str, str, int]],
    image_logits: dict[str, np.ndarray],
    audio_logits: dict[str, np.ndarray],
) -> float:
    best_temperature = 1.0
    best_nll = float("inf")
    for step in range(5, 41):
        temperature = step / 10
        _accuracy, nll, _, _ = score(system, module, pairs, image_logits, audio_logits, temperature)
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    print(f"temperature {best_temperature:.1f} calib_nll={best_nll:.3f}", flush=True)
    return best_temperature


def write_results(rows: list[dict[str, str]]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "system", "accuracy", "nll", "n", "temperature", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_species(labels: list[int], predictions: list[int], destination: Path = SPECIES_PATH) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["common_name", "n", "n_correct", "accuracy"],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, name in enumerate(LABELS):
            chosen = [pred for label, pred in zip(labels, predictions) if label == index]
            correct = sum(pred == index for pred in chosen)
            count = len(chosen)
            accuracy = correct / count if count else 0.0
            writer.writerow(
                {
                    "common_name": name,
                    "n": str(count),
                    "n_correct": str(correct),
                    "accuracy": f"{accuracy:.6f}",
                }
            )


def load_logit_cache(modality: str, split_name: str) -> dict[str, np.ndarray]:
    cache_path = LOGIT_DIR / f"{modality}_{split_name}.npz"
    if not cache_path.is_file():
        raise RuntimeError(f"missing cached logits: {cache_path.name}")
    stored = np.load(cache_path, allow_pickle=False)
    return {
        str(sample_id): np.asarray(logits, dtype=np.float32)
        for sample_id, logits in zip(stored["ids"], stored["logits"])
    }


def rows_for_cache(modality: str, split_name: str, cache: dict[str, np.ndarray]) -> list[dict[str, str]]:
    return [row for row in read_split(modality, split_name) if row["sample_id"] in cache]


def mean_window_path(split_name: str, directory: Path = LOGIT_DIR) -> Path:
    return directory / f"audio_mean_windows_{split_name}.npz"


def load_mean_window_logits(split_name: str, directory: Path = LOGIT_DIR) -> dict[str, tuple[np.ndarray, int]]:
    cache_path = mean_window_path(split_name, directory)
    if not cache_path.is_file():
        return {}
    stored = np.load(cache_path, allow_pickle=False)
    return {
        str(sample_id): (np.asarray(logits, dtype=np.float32), int(count))
        for sample_id, logits, count in zip(stored["ids"], stored["logits"], stored["n_windows"])
    }


def save_mean_window_logits(
    split_name: str,
    logits: dict[str, np.ndarray],
    counts: dict[str, int],
    order: list[str],
    directory: Path = LOGIT_DIR,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        mean_window_path(split_name, directory),
        ids=np.array(order),
        logits=np.stack([logits[sample_id] for sample_id in order]).astype(np.float32),
        n_windows=np.array([counts[sample_id] for sample_id in order], dtype=np.int32),
    )


def read_result_rows() -> list[dict[str, str]]:
    if not RESULT_PATH.is_file():
        return []
    with RESULT_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def result_exists(rows: list[dict[str, str]], stage: str, system: str) -> bool:
    return any(row["stage"] == stage and row["system"] == system for row in rows)


def result_row(
    stage: str,
    accuracy: float,
    nll: float,
    count: int,
    temperature: float,
    note: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "system": WINDOW_SYSTEM,
        "accuracy": f"{accuracy:.6f}",
        "nll": f"{nll:.6f}",
        "n": str(count),
        "temperature": f"{temperature:.1f}",
        "note": note,
    }


def mean_windows_for_split(
    model: nn.Module,
    split_name: str,
    fallback: dict[str, np.ndarray],
    directory: Path = LOGIT_DIR,
) -> dict[str, np.ndarray]:
    rows = rows_for_cache("audio", split_name, fallback)
    order = [row["sample_id"] for row in rows]
    cached = load_mean_window_logits(split_name, directory)
    if cached and set(order) <= set(cached):
        print(f"loaded window cache {split_name} {len(order)}", flush=True)
        return {sample_id: cached[sample_id][0] for sample_id in order}

    logits: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        try:
            averaged, count = train.audio_window_logits(model, row)
        except Exception as exc:
            print(f"window fallback {row['file_path']}: {exc}", flush=True)
            averaged = np.asarray(fallback[row["sample_id"]], dtype=np.float32)
            count = 1
        logits[row["sample_id"]] = averaged
        counts[row["sample_id"]] = count
        if index % 25 == 0 or index == len(rows):
            print(f"windows {split_name} {index}/{len(rows)} windows_used {count}", flush=True)
    save_mean_window_logits(split_name, logits, counts, order, directory)
    return logits


def check_audio_checkpoint(model: nn.Module, row: dict[str, str], cached: np.ndarray) -> None:
    fresh, _label = MelImageSet([row], train=False)[0]
    with torch.no_grad():
        logits = model(fresh.unsqueeze(0)).numpy()[0]
    delta = float(np.max(np.abs(logits - cached)))
    print(f"checkpoint check {row['sample_id']} max_abs={delta:.5f}", flush=True)
    if delta > 0.05:
        raise RuntimeError("the saved recording model does not match the cached first-window logits")


def evaluate_windows() -> None:
    """Score whole-file recording windows. Touch the test split only if val improves."""
    checkpoint_path = ROOT / "models" / "fusion_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("audio_pooling") == "mean_windows":
        print("windows already adopted; test split will not be scored again", flush=True)
        return
    if checkpoint.get("system") != "linear":
        raise RuntimeError("window pooling stays on the saved linear fusion layer")
    existing = read_result_rows()
    if result_exists(existing, "test", WINDOW_SYSTEM) or result_exists(existing, "val", WINDOW_SYSTEM):
        print("window result already recorded; test split will not be scored again", flush=True)
        return

    audio_checkpoint = torch.load(ROOT / "models" / "audio_best.pt", map_location="cpu", weights_only=False)
    if audio_checkpoint.get("labels") != LABELS:
        raise RuntimeError("audio checkpoint labels do not match the species list")
    audio_net = audio_mobilenet()
    audio_net.load_state_dict(audio_checkpoint["state_dict"])
    audio_net.eval()

    image_val = load_logit_cache("image", "val")
    audio_val = load_logit_cache("audio", "val")
    image_rows = rows_for_cache("image", "val", image_val)
    audio_rows = rows_for_cache("audio", "val", audio_val)
    check_audio_checkpoint(audio_net, audio_rows[0], audio_val[audio_rows[0]["sample_id"]])

    window_val = mean_windows_for_split(audio_net, "val", audio_val)
    val_pairs = build_pairs("val", image_val, window_val, image_rows, audio_rows, exhaustive=True)
    module = LogitFuse()
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    accuracy, nll, _, _ = score("linear", module, val_pairs, image_val, window_val)
    print(
        f"WINDOW_VAL acc={accuracy:.6f} nll={nll:.6f} n={len(val_pairs)} floor={WINDOW_VAL_FLOOR:.6f}",
        flush=True,
    )
    if accuracy <= WINDOW_VAL_FLOOR:
        existing.append(
            result_row(
                "val",
                accuracy,
                nll,
                len(val_pairs),
                1.0,
                f"not adopted; did not beat linear {WINDOW_VAL_FLOOR:.3f}",
            )
        )
        write_results(existing)
        print("WINDOW_KEEP first 3 seconds", flush=True)
        return

    image_calib = load_logit_cache("image", "calib")
    audio_calib = load_logit_cache("audio", "calib")
    window_calib = mean_windows_for_split(audio_net, "calib", audio_calib)
    calib_pairs = build_pairs(
        "calib",
        image_calib,
        window_calib,
        rows_for_cache("image", "calib", image_calib),
        rows_for_cache("audio", "calib", audio_calib),
        exhaustive=True,
    )
    temperature = fit_temperature("linear", module, calib_pairs, image_calib, window_calib)
    calib_accuracy, calib_nll, _, _ = score(
        "linear", module, calib_pairs, image_calib, window_calib, temperature
    )

    image_test = load_logit_cache("image", "test")
    audio_test = load_logit_cache("audio", "test")
    window_test = mean_windows_for_split(audio_net, "test", audio_test)
    test_pairs = build_pairs(
        "test",
        image_test,
        window_test,
        rows_for_cache("image", "test", image_test),
        rows_for_cache("audio", "test", audio_test),
        exhaustive=True,
    )
    test_accuracy, test_nll, test_labels, test_predictions = score(
        "linear", module, test_pairs, image_test, window_test, temperature
    )
    print(
        f"WINDOW_TEST acc={test_accuracy:.6f} nll={test_nll:.6f} "
        f"temperature={temperature:.1f} n={len(test_pairs)}",
        flush=True,
    )

    for row in existing:
        if row["stage"] == "test" and row["system"] == "linear" and "superseded" not in row["note"]:
            row["note"] = f"{row['note']}; superseded by {WINDOW_SYSTEM}".strip("; ")
    window_note = f"adopted; mean of up to {train.MAX_AUDIO_WINDOWS} windows of {train.CLIP_SECONDS:.0f} seconds"
    existing.append(result_row("val", accuracy, nll, len(val_pairs), 1.0, window_note))
    existing.append(
        result_row("calib", calib_accuracy, calib_nll, len(calib_pairs), temperature, "temperature fit on calib")
    )
    existing.append(result_row("test", test_accuracy, test_nll, len(test_pairs), temperature, "scored once"))
    archive = SPECIES_PATH.with_name("fusion_test_by_species_first_window.csv")
    if SPECIES_PATH.is_file() and not archive.is_file():
        shutil.copyfile(SPECIES_PATH, archive)
    write_results(existing)
    write_species(test_labels, test_predictions)
    checkpoint["temperature"] = float(temperature)
    checkpoint["audio_pooling"] = "mean_windows"
    checkpoint["max_windows"] = train.MAX_AUDIO_WINDOWS
    checkpoint["window_seconds"] = train.CLIP_SECONDS
    torch.save(checkpoint, checkpoint_path)
    print("WINDOW_ADOPTED", flush=True)


def main() -> None:
    torch.manual_seed(SEED)
    image_model_net, audio_model_net = load_models()
    image_rows = {split_name: read_split("image", split_name) for split_name in ("train", "val", "calib", "test")}
    audio_rows = {split_name: usable_audio(split_name) for split_name in ("train", "val", "calib", "test")}
    image_logits = {
        split_name: embed_split(image_model_net, image_rows[split_name], "image", split_name)
        for split_name in image_rows
    }
    audio_logits = {
        split_name: embed_split(audio_model_net, audio_rows[split_name], "audio", split_name)
        for split_name in audio_rows
    }
    pair_sets = {
        split_name: build_pairs(
            split_name,
            image_logits[split_name],
            audio_logits[split_name],
            image_rows[split_name],
            audio_rows[split_name],
            exhaustive=split_name != "train",
        )
        for split_name in image_rows
    }

    weighted = train_module("weighted", WeightedSum(), pair_sets["train"], pair_sets["val"], image_logits, audio_logits)
    linear = train_module("linear", LogitFuse(), pair_sets["train"], pair_sets["val"], image_logits, audio_logits)
    candidates: dict[str, nn.Module | None] = {
        "image_only": None,
        "mean_prob": None,
        "weighted": weighted,
        "linear": linear,
    }
    val_scores: dict[str, tuple[float, float]] = {}
    for name, module in candidates.items():
        accuracy, nll, _, _ = score(name, module, pair_sets["val"], image_logits["val"], audio_logits["val"])
        val_scores[name] = (accuracy, nll)
        print(f"val {name} acc={accuracy:.3f} nll={nll:.3f}", flush=True)
    chosen = max(val_scores, key=lambda name: (val_scores[name][0], -val_scores[name][1]))
    chosen_module = candidates[chosen]
    print(f"selected {chosen}", flush=True)
    if chosen == "weighted" and isinstance(chosen_module, WeightedSum):
        image_weight, audio_weight = chosen_module.weights()
        print(f"weights image={image_weight:.3f} audio={audio_weight:.3f}", flush=True)

    temperature = fit_temperature(
        chosen,
        chosen_module,
        pair_sets["calib"],
        image_logits["calib"],
        audio_logits["calib"],
    )
    calib_accuracy, calib_nll, _, _ = score(
        chosen,
        chosen_module,
        pair_sets["calib"],
        image_logits["calib"],
        audio_logits["calib"],
        temperature,
    )
    test_accuracy, test_nll, test_labels, test_predictions = score(
        chosen,
        chosen_module,
        pair_sets["test"],
        image_logits["test"],
        audio_logits["test"],
        temperature,
    )
    print(
        f"TEST {chosen} acc={test_accuracy:.3f} nll={test_nll:.3f} "
        f"temperature={temperature:.1f} n={len(pair_sets['test'])}",
        flush=True,
    )

    result_rows = []
    for name, (accuracy, nll) in val_scores.items():
        result_rows.append(
            {
                "stage": "val",
                "system": name,
                "accuracy": f"{accuracy:.6f}",
                "nll": f"{nll:.6f}",
                "n": str(len(pair_sets["val"])),
                "temperature": "1.0",
                "note": "selected" if name == chosen else "",
            }
        )
    result_rows.append(
        {
            "stage": "calib",
            "system": chosen,
            "accuracy": f"{calib_accuracy:.6f}",
            "nll": f"{calib_nll:.6f}",
            "n": str(len(pair_sets["calib"])),
            "temperature": f"{temperature:.1f}",
            "note": "temperature fit on calib",
        }
    )
    result_rows.append(
        {
            "stage": "test",
            "system": chosen,
            "accuracy": f"{test_accuracy:.6f}",
            "nll": f"{test_nll:.6f}",
            "n": str(len(pair_sets["test"])),
            "temperature": f"{temperature:.1f}",
            "note": "scored once",
        }
    )
    write_results(result_rows)
    write_species(test_labels, test_predictions)
    if chosen == "weighted" and isinstance(chosen_module, WeightedSum):
        torch.save({"state_dict": chosen_module.state_dict(), "system": chosen, "temperature": temperature}, ROOT / "models" / "fusion_best.pt")
    elif isinstance(chosen_module, LogitFuse):
        torch.save({"state_dict": chosen_module.state_dict(), "system": chosen, "temperature": temperature}, ROOT / "models" / "fusion_best.pt")
    else:
        torch.save({"system": chosen, "temperature": temperature}, ROOT / "models" / "fusion_best.pt")
    print("FUSION_DONE", flush=True)


def adopted_val_floor(rows: list[dict[str, str]]) -> float:
    for row in reversed(rows):
        if row["stage"] == "val" and str(row.get("note", "")).startswith("adopted"):
            return float(row["accuracy"])
    for row in rows:
        if row["stage"] == "val" and row["system"] == "linear":
            return float(row["accuracy"])
    return WINDOW_VAL_FLOOR


AUG_DIR = ROOT / "data" / "cache" / "logits_aug"
BACKUP_LOGIT_DIR = LOGIT_DIR / "backup_before_aug"
AUG_SPECIES_PATH = SPECIES_PATH.with_name("fusion_test_by_species_aug.csv")
SPLIT_NAMES = ("train", "val", "calib", "test")
POOLINGS = ("first_window", "mean_windows")


def load_candidate(kind: str, changed: bool) -> nn.Module:
    """The retrained model when it was kept, otherwise the saved one."""
    stem = "image" if kind == "image" else "audio"
    path = ROOT / "models" / (f"{stem}_aug.pt" if changed else f"{stem}_best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != LABELS:
        raise RuntimeError(f"{path.name} labels do not match the species list")
    net = image_model(pretrained=False) if kind == "image" else audio_mobilenet(pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    print(f"using {path.name}", flush=True)
    return net


def seed_unchanged(image_changed: bool, audio_changed: bool) -> None:
    """A model that was not retrained keeps the logits already cached for it."""
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for split_name in SPLIT_NAMES:
        if not image_changed:
            names.append(f"image_{split_name}.npz")
        if not audio_changed:
            names.append(f"audio_{split_name}.npz")
            names.append(f"audio_mean_windows_{split_name}.npz")
    for name in names:
        source = LOGIT_DIR / name
        target = AUG_DIR / name
        if source.is_file() and not target.is_file():
            shutil.copyfile(source, target)


def selection_run(
    pooling: str,
    image_logits: dict[str, dict[str, np.ndarray]],
    audio_logits: dict[str, dict[str, np.ndarray]],
    image_rows: dict[str, list[dict[str, str]]],
    audio_rows_: dict[str, list[dict[str, str]]],
) -> dict:
    """Fit on train, choose on val. Calib and test are not touched here."""
    print(f"selection with audio={pooling}", flush=True)
    pairs = {
        split_name: build_pairs(
            split_name,
            image_logits[split_name],
            audio_logits[split_name],
            image_rows[split_name],
            audio_rows_[split_name],
            exhaustive=split_name != "train",
        )
        for split_name in ("train", "val")
    }
    weighted = train_module("weighted", WeightedSum(), pairs["train"], pairs["val"], image_logits, audio_logits)
    linear = train_module("linear", LogitFuse(), pairs["train"], pairs["val"], image_logits, audio_logits)
    modules: dict[str, nn.Module | None] = {
        "image_only": None,
        "audio_only": None,
        "mean_prob": None,
        "weighted": weighted,
        "linear": linear,
    }
    val_scores: dict[str, tuple[float, float]] = {}
    for name, module in modules.items():
        accuracy, nll, _, _ = score(name, module, pairs["val"], image_logits["val"], audio_logits["val"])
        val_scores[name] = (accuracy, nll)
        print(f"val {pooling} {name} acc={accuracy:.4f} nll={nll:.4f}", flush=True)
    chosen = max(
        ("image_only", "mean_prob", "weighted", "linear"),
        key=lambda name: (val_scores[name][0], -val_scores[name][1]),
    )
    print(f"selected {chosen} with audio={pooling} val acc={val_scores[chosen][0]:.4f}", flush=True)
    return {
        "pooling": pooling,
        "modules": modules,
        "val_scores": val_scores,
        "val_pairs": len(pairs["val"]),
        "chosen": chosen,
    }


def aug_row(stage: str, system: str, accuracy: float, nll: float, count: int, temperature: float, note: str) -> dict[str, str]:
    return {
        "stage": stage,
        "system": f"{system}_aug",
        "accuracy": f"{accuracy:.6f}",
        "nll": f"{nll:.6f}",
        "n": str(count),
        "temperature": f"{temperature:.1f}",
        "note": note,
    }


def rebuild(image_changed: bool, audio_changed: bool) -> None:
    """Refit fusion for retrained checkpoints, then score test once if val clears the floor."""
    if not (image_changed or audio_changed):
        raise RuntimeError("nothing was retrained; keep the saved fusion layer")
    torch.manual_seed(SEED)
    existing = read_result_rows()
    if any(row["system"].endswith("_aug") for row in existing):
        print("an _aug result is already recorded; test split will not be scored again", flush=True)
        return
    floor = adopted_val_floor(existing)
    print(f"fused val floor {floor:.6f}", flush=True)

    image_net = load_candidate("image", image_changed)
    audio_net = load_candidate("audio", audio_changed)
    seed_unchanged(image_changed, audio_changed)

    image_rows = {split_name: read_split("image", split_name) for split_name in ("train", "val")}
    audio_rows_ = {split_name: usable_audio(split_name) for split_name in ("train", "val")}
    image_logits = {
        split_name: embed_split(image_net, image_rows[split_name], "image", split_name, AUG_DIR)
        for split_name in image_rows
    }
    first_logits = {
        split_name: embed_split(audio_net, audio_rows_[split_name], "audio", split_name, AUG_DIR)
        for split_name in audio_rows_
    }
    window_logits = {
        split_name: mean_windows_for_split(audio_net, split_name, first_logits[split_name], AUG_DIR)
        for split_name in audio_rows_
    }
    by_pooling = {"first_window": first_logits, "mean_windows": window_logits}
    runs = {
        pooling: selection_run(pooling, image_logits, by_pooling[pooling], image_rows, audio_rows_)
        for pooling in POOLINGS
    }
    first_best = runs["first_window"]["val_scores"][runs["first_window"]["chosen"]][0]
    window_best = runs["mean_windows"]["val_scores"][runs["mean_windows"]["chosen"]][0]
    pooling = "mean_windows" if window_best > first_best else "first_window"
    other = "first_window" if pooling == "mean_windows" else "mean_windows"
    run = runs[pooling]
    chosen = run["chosen"]
    accuracy, nll = run["val_scores"][chosen]
    print(
        f"WINDOWS_VS_SINGLE val single={first_best:.4f} windows={window_best:.4f} keep={pooling}",
        flush=True,
    )

    def val_rows_for(run_: dict, adopted: bool, why: str) -> list[dict[str, str]]:
        rows = []
        for name, (acc, nll_) in run_["val_scores"].items():
            note = f"audio={run_['pooling']}"
            if name == run_["chosen"]:
                note = f"{'adopted; selected' if adopted else 'not adopted; selected'}; audio={run_['pooling']}"
                if why:
                    note += f"; {why}"
            rows.append(aug_row("val", name, acc, nll_, run_["val_pairs"], 1.0, note))
        return rows

    other_run = runs[other]
    other_acc, other_nll = other_run["val_scores"][other_run["chosen"]]
    comparison = aug_row(
        "val",
        f"{other_run['chosen']}_{other}",
        other_acc,
        other_nll,
        other_run["val_pairs"],
        1.0,
        "comparison only; not used",
    )

    if chosen != "linear" or accuracy <= floor:
        why = (
            f"did not beat {floor:.3f}"
            if accuracy <= floor
            else "the dashboard serves linear fusion only"
        )
        write_results(existing + val_rows_for(run, False, why) + [comparison])
        print(f"FUSION_KEEP_OLD val {accuracy:.6f} floor {floor:.6f} ({why}); test split not scored", flush=True)
        return

    image_rows["calib"] = read_split("image", "calib")
    image_rows["test"] = read_split("image", "test")
    audio_rows_["calib"] = usable_audio("calib")
    audio_rows_["test"] = usable_audio("test")
    for split_name in ("calib", "test"):
        image_logits[split_name] = embed_split(image_net, image_rows[split_name], "image", split_name, AUG_DIR)
        first_logits[split_name] = embed_split(audio_net, audio_rows_[split_name], "audio", split_name, AUG_DIR)
        window_logits[split_name] = mean_windows_for_split(
            audio_net, split_name, first_logits[split_name], AUG_DIR
        )
    audio_logits = by_pooling[pooling]
    module = run["modules"][chosen]
    pairs = {
        split_name: build_pairs(
            split_name,
            image_logits[split_name],
            audio_logits[split_name],
            image_rows[split_name],
            audio_rows_[split_name],
            exhaustive=True,
        )
        for split_name in ("calib", "test")
    }
    temperature = fit_temperature(chosen, module, pairs["calib"], image_logits["calib"], audio_logits["calib"])
    calib_accuracy, calib_nll, _, _ = score(
        chosen, module, pairs["calib"], image_logits["calib"], audio_logits["calib"], temperature
    )
    test_scores: dict[str, tuple[float, float]] = {}
    test_labels: list[int] = []
    test_predictions: list[int] = []
    for name in ("image_only", "audio_only", "mean_prob"):
        acc, nll_, _, _ = score(name, None, pairs["test"], image_logits["test"], audio_logits["test"])
        test_scores[name] = (acc, nll_)
    test_accuracy, test_nll, test_labels, test_predictions = score(
        chosen, module, pairs["test"], image_logits["test"], audio_logits["test"], temperature
    )
    test_scores[chosen] = (test_accuracy, test_nll)
    print(
        f"TEST {chosen} audio={pooling} acc={test_accuracy:.6f} nll={test_nll:.6f} "
        f"temperature={temperature:.1f} n={len(pairs['test'])}",
        flush=True,
    )

    note = f"audio={pooling}"
    new_rows = val_rows_for(run, True, "")
    new_rows.append(comparison)
    new_rows.append(
        aug_row("calib", chosen, calib_accuracy, calib_nll, len(pairs["calib"]), temperature, f"adopted; temperature fit on calib; {note}")
    )
    for name, (acc, nll_) in test_scores.items():
        label = f"adopted; scored once; {note}" if name == chosen else f"scored once; {note}"
        test_temperature = temperature if name == chosen else 1.0
        new_rows.append(aug_row("test", name, acc, nll_, len(pairs["test"]), test_temperature, label))

    install_aug(
        image_changed,
        audio_changed,
        module,
        float(temperature),
        pooling,
        existing + new_rows,
        test_labels,
        test_predictions,
    )
    print("FUSION_ADOPTED", flush=True)


def install_aug(
    image_changed: bool,
    audio_changed: bool,
    module: nn.Module,
    temperature: float,
    pooling: str,
    rows: list[dict[str, str]],
    test_labels: list[int],
    test_predictions: list[int],
) -> None:
    """Make the retrained checkpoints, logit caches, and fusion layer the live ones."""
    for kind, changed in (("image", image_changed), ("audio", audio_changed)):
        if changed:
            shutil.copyfile(ROOT / "models" / f"{kind}_aug.pt", ROOT / "models" / f"{kind}_best.pt")
    BACKUP_LOGIT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(AUG_DIR.glob("*.npz")):
        target = LOGIT_DIR / source.name
        saved = BACKUP_LOGIT_DIR / source.name
        if target.is_file() and not saved.is_file():
            shutil.copyfile(target, saved)
        shutil.copyfile(source, target)
    write_species(test_labels, test_predictions, AUG_SPECIES_PATH)
    write_species(test_labels, test_predictions)
    write_results(rows)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "system": "linear",
            "temperature": temperature,
            "audio_pooling": pooling,
            "max_windows": train.MAX_AUDIO_WINDOWS,
            "window_seconds": train.CLIP_SECONDS,
        },
        ROOT / "models" / "fusion_best.pt",
    )


REJECT_PATH = ROOT / "models" / "reject.json"
REJECT_KEEP = 0.95


def fused_confidence(
    split_name: str, module: nn.Module, temperature: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused confidence, photo confidence, and correctness for every same-species pair of a split."""
    images = load_logit_cache("image", split_name)
    audio = {sample_id: logits for sample_id, (logits, _count) in load_mean_window_logits(split_name).items()}
    pairs = build_pairs(
        split_name,
        images,
        audio,
        rows_for_cache("image", split_name, images),
        rows_for_cache("audio", split_name, audio),
        exhaustive=True,
    )
    loader = DataLoader(PairSet(pairs, images, audio), batch_size=256, shuffle=False)
    confidence: list[float] = []
    photo: list[float] = []
    correct: list[bool] = []
    with torch.no_grad():
        for image_batch, audio_batch, target in loader:
            photo.extend(torch.softmax(image_batch, dim=1).max(dim=1).values.tolist())
            probabilities = torch.softmax(module(image_batch, audio_batch) / temperature, dim=1)
            top, index = probabilities.max(dim=1)
            confidence.extend(top.tolist())
            correct.extend((index == target).tolist())
    return np.array(confidence), np.array(photo), np.array(correct)


def fit_reject() -> None:
    """Choose the confidence cuts below which the page says the species is not in the data.

    Every calib pair is one of the 20 species, so each cut keeps REJECT_KEEP of them.
    Two cuts: the highest fused probability, and the highest photo-only probability.
    The recording model is too uncertain on its own to gate anything.
    Test is not read. Val only reports how the cuts behave on pairs they were not fit on.
    """
    checkpoint = torch.load(ROOT / "models" / "fusion_best.pt", map_location="cpu", weights_only=False)
    module = LogitFuse()
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    temperature = float(checkpoint["temperature"])
    calib_fused, calib_photo, _ = fused_confidence("calib", module, temperature)
    fused_cut = float(np.quantile(calib_fused, 1.0 - REJECT_KEEP))
    photo_cut = float(np.quantile(calib_photo, 1.0 - REJECT_KEEP))
    val_fused, val_photo, val_correct = fused_confidence("val", module, temperature)
    accepted = (val_fused >= fused_cut) & (val_photo >= photo_cut)
    summary = {
        "metric": "highest fused probability and highest photo-only probability",
        "threshold": round(fused_cut, 4),
        "photo_threshold": round(photo_cut, 4),
        "kept_of_known_species_each_cut": REJECT_KEEP,
        "fit_on": f"calib, {len(calib_fused)} pairs from the 20 species",
        "val_accepted_both_cuts": round(float(accepted.mean()), 4),
        "val_accuracy_all": round(float(val_correct.mean()), 4),
        "val_accuracy_accepted": round(float(val_correct[accepted].mean()), 4),
        "val_accuracy_rejected": round(float(val_correct[~accepted].mean()), 4) if (~accepted).any() else None,
        "note": "No bird outside the 20 species exists in the data, so the rate of catching one is not measured.",
    }
    REJECT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reject":
        fit_reject()
    elif len(sys.argv) > 1 and sys.argv[1] == "windows":
        evaluate_windows()
    elif len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild("image" in sys.argv[2:], "audio" in sys.argv[2:])
    else:
        main()
