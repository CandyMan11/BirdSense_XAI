"""Fuse the photo and recording models.

Photos and recordings come from different observations, so a training pair is
one photo and one recording of the same species from the same split.
The fusion rule is chosen on val. Temperature is fit on calib.
The test split is scored once, after that choice is fixed.
"""

from __future__ import annotations

import csv
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
def embed_split(model: nn.Module, rows: list[dict[str, str]], modality: str, split_name: str) -> dict[str, np.ndarray]:
    cache_path = LOGIT_DIR / f"{modality}_{split_name}.npz"
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
    LOGIT_DIR.mkdir(parents=True, exist_ok=True)
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


def write_species(labels: list[int], predictions: list[int]) -> None:
    with SPECIES_PATH.open("w", encoding="utf-8", newline="") as handle:
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


if __name__ == "__main__":
    main()
