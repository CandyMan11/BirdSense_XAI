"""Local dashboard for one photo, one recording, and the fused species call.

This serves inference only. It does not train, and it does not rescore the test split.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F

from download_dataset import SPECIES
from fuse import LogitFuse
from train import LABELS, LABEL_TO_INDEX, ROOT, ImageSet, MelImageSet, audio_mobilenet, image_model

PORT = 8765
RAW_ROOT = (ROOT / "data" / "raw").resolve()
SPLITS_PATH = ROOT / "data" / "metadata" / "splits.csv"
SKIP_PATH = ROOT / "data" / "metadata" / "audio_decode_skips.csv"
RESULTS_PATH = ROOT / "data" / "metadata" / "fusion_results.csv"
PAGE_PATH = ROOT / "dashboard" / "index.html"

SCIENTIFIC = {common: scientific for common, scientific in SPECIES}
LOCK = threading.Lock()
CATALOG: list[dict[str, str]] = []
BY_ID: dict[str, dict[str, str]] = {}
ALLOWED: set[str] = set()
SKIPPED: set[str] = set()
IMAGE_NET = None
AUDIO_NET = None
FUSION_NET = None
TEMPERATURE = 1.0


def load_catalog() -> None:
    if SKIP_PATH.is_file():
        with SKIP_PATH.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                SKIPPED.add(row["sample_id"])
    with SPLITS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = {
                "sample_id": row["sample_id"],
                "file_path": row["file_path"],
                "common_name": row["common_name"],
                "scientific_name": row["scientific_name"],
                "modality": row["modality"],
                "split": row["split"],
                "decodable": "no" if row["sample_id"] in SKIPPED else "yes",
            }
            CATALOG.append(item)
            BY_ID[item["sample_id"]] = item
            ALLOWED.add(item["file_path"].replace("\\", "/"))


def load_models() -> None:
    global IMAGE_NET, AUDIO_NET, FUSION_NET, TEMPERATURE
    IMAGE_NET = image_model()
    image_checkpoint = torch.load(ROOT / "models" / "image_best.pt", map_location="cpu", weights_only=False)
    IMAGE_NET.load_state_dict(image_checkpoint["state_dict"])
    IMAGE_NET.eval()

    AUDIO_NET = audio_mobilenet()
    audio_checkpoint = torch.load(ROOT / "models" / "audio_best.pt", map_location="cpu", weights_only=False)
    AUDIO_NET.load_state_dict(audio_checkpoint["state_dict"])
    AUDIO_NET.eval()

    FUSION_NET = LogitFuse()
    fusion_checkpoint = torch.load(ROOT / "models" / "fusion_best.pt", map_location="cpu", weights_only=False)
    FUSION_NET.load_state_dict(fusion_checkpoint["state_dict"])
    FUSION_NET.eval()
    TEMPERATURE = float(fusion_checkpoint["temperature"])


def recorded_summary() -> dict[str, str]:
    summary = {"test_accuracy": "", "val_fusion": "", "val_image": "", "temperature": f"{TEMPERATURE:.1f}"}
    if not RESULTS_PATH.is_file():
        return summary
    with RESULTS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["stage"] == "test":
                summary["test_accuracy"] = row["accuracy"]
            elif row["stage"] == "val" and row["system"] == "linear":
                summary["val_fusion"] = row["accuracy"]
            elif row["stage"] == "val" and row["system"] == "image_only":
                summary["val_image"] = row["accuracy"]
    return summary


def safe_file(relative: str) -> Path:
    relative = relative.replace("\\", "/").lstrip("/")
    if relative not in ALLOWED:
        raise ValueError("file is not in the dataset catalog")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(RAW_ROOT):
        raise ValueError("file is outside the raw dataset")
    return path


def top_call(probabilities: torch.Tensor) -> dict[str, object]:
    index = int(probabilities.argmax().item())
    return {
        "species": LABELS[index],
        "scientific": SCIENTIFIC[LABELS[index]],
        "probability": float(probabilities[index].item()),
        "index": index,
    }


def ranking(probabilities: torch.Tensor, limit: int = 5) -> list[dict[str, object]]:
    scores, indices = torch.topk(probabilities, k=min(limit, probabilities.shape[0]))
    return [
        {
            "species": LABELS[int(index)],
            "scientific": SCIENTIFIC[LABELS[int(index)]],
            "probability": float(score),
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]


@torch.no_grad()
def predict(image_id: str, audio_id: str) -> dict[str, object]:
    image_row = BY_ID.get(image_id)
    audio_row = BY_ID.get(audio_id)
    if image_row is None or image_row["modality"] != "image":
        raise ValueError("choose a photo from the catalog")
    if audio_row is None or audio_row["modality"] != "audio":
        raise ValueError("choose a recording from the catalog")
    if audio_row["decodable"] != "yes":
        raise ValueError("that recording could not be decoded")

    image_tensor, _label = ImageSet([image_row], train=False)[0]
    audio_tensor, _label = MelImageSet([audio_row], train=False)[0]
    image_logits = IMAGE_NET(image_tensor.unsqueeze(0))
    audio_logits = AUDIO_NET(audio_tensor.unsqueeze(0))
    fused_logits = FUSION_NET(image_logits, audio_logits)
    fused_probabilities = torch.softmax(fused_logits / TEMPERATURE, dim=1)[0]
    image_probabilities = torch.softmax(image_logits, dim=1)[0]
    audio_probabilities = torch.softmax(audio_logits, dim=1)[0]
    fused = top_call(fused_probabilities)
    winner = int(fused["index"])
    weight = FUSION_NET.fc.weight.detach()[winner]
    image_part = float(torch.dot(weight[: len(LABELS)], image_logits[0]).item())
    audio_part = float(torch.dot(weight[len(LABELS) :], audio_logits[0]).item())
    bias = float(FUSION_NET.fc.bias.detach()[winner].item())
    return {
        "fused": fused,
        "image_only": top_call(image_probabilities),
        "audio_only": top_call(audio_probabilities),
        "ranking": ranking(fused_probabilities),
        "contribution": {"image": image_part, "audio": audio_part, "bias": bias},
        "temperature": TEMPERATURE,
        "image_split": image_row["split"],
        "audio_split": audio_row["split"],
        "same_species_files": image_row["common_name"] == audio_row["common_name"],
        "file_species": {
            "image": image_row["common_name"],
            "audio": audio_row["common_name"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/catalog":
            self.send_json(
                {
                    "species": [{"common": common, "scientific": scientific} for common, scientific in SPECIES],
                    "files": CATALOG,
                }
            )
            return
        if parsed.path == "/api/summary":
            self.send_json(recorded_summary())
            return
        if parsed.path == "/file":
            relative = parse_qs(parsed.query).get("path", [""])[0]
            try:
                path = safe_file(relative)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with LOCK:
                result = predict(str(payload.get("image_id", "")), str(payload.get("audio_id", "")))
        except (json.JSONDecodeError, ValueError, OSError, RuntimeError) as exc:
            self.send_json({"error": str(exc)}, 400)
            return
        self.send_json(result)


def main() -> None:
    print("loading catalog and models", flush=True)
    load_catalog()
    load_models()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"dashboard http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
