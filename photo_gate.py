"""Photo gate: is this photo one of the 20 species, or something else?

The photo model has no "none of these" answer, so a bald eagle still gets a
species name. The gate is a small logistic regression on the photo model's
pooled features. It learns "known" from the train photos and "unknown" from
photos of other species (data/ood, gitignored). Calib sets the cut so that
KEEP of the known photos pass. Val reports how it behaves on photos it did
not see. Test is not read.

    python photo_gate.py download   # fetch photos of species outside the 20
    python photo_gate.py fit        # fit the gate and write models/photo_gate.npz
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

from download_dataset import INAT_API, USER_AGENT
from train import LABEL_TO_INDEX, ROOT, image_tensor_from_path, read_split

OOD_DIR = ROOT / "data" / "ood"
GATE_PATH = ROOT / "models" / "photo_gate.npz"
REPORT_PATH = ROOT / "models" / "photo_gate.json"
KEEP = 0.90
PER_TAXON = 8

# Species outside the 20, plus a few non-birds.
OTHER_TAXA = [
    ("bald_eagle", "Haliaeetus leucocephalus"), ("osprey", "Pandion haliaetus"),
    ("great_blue_heron", "Ardea herodias"), ("wild_turkey", "Meleagris gallopavo"),
    ("barn_owl", "Tyto alba"), ("red_tailed_hawk", "Buteo jamaicensis"),
    ("belted_kingfisher", "Megaceryle alcyon"), ("ruby_throated_hummingbird", "Archilochus colubris"),
    ("northern_flicker", "Colaptes auratus"), ("great_horned_owl", "Bubo virginianus"),
    ("turkey_vulture", "Cathartes aura"), ("coopers_hawk", "Accipiter cooperii"),
    ("peregrine_falcon", "Falco peregrinus"), ("american_kestrel", "Falco sparverius"),
    ("eastern_screech_owl", "Megascops asio"), ("barred_owl", "Strix varia"),
    ("great_egret", "Ardea alba"), ("green_heron", "Butorides virescens"),
    ("double_crested_cormorant", "Nannopterus auritus"), ("herring_gull", "Larus argentatus"),
    ("ring_billed_gull", "Larus delawarensis"), ("killdeer", "Charadrius vociferus"),
    ("sandhill_crane", "Antigone canadensis"), ("brown_pelican", "Pelecanus occidentalis"),
    ("mute_swan", "Cygnus olor"), ("bufflehead", "Bucephala albeola"),
    ("hooded_merganser", "Lophodytes cucullatus"), ("rock_pigeon", "Columba livia"),
    ("pileated_woodpecker", "Dryocopus pileatus"), ("northern_mockingbird", "Mimus polyglottos"),
    ("brown_thrasher", "Toxostoma rufum"), ("eastern_bluebird", "Sialia sialis"),
    ("cedar_waxwing", "Bombycilla cedrorum"), ("european_starling", "Sturnus vulgaris"),
    ("yellow_rumped_warbler", "Setophaga coronata"), ("dark_eyed_junco", "Junco hyemalis"),
    ("white_throated_sparrow", "Zonotrichia albicollis"), ("scarlet_tanager", "Piranga olivacea"),
    ("indigo_bunting", "Passerina cyanea"), ("house_finch", "Haemorhous mexicanus"),
    ("carolina_wren", "Thryothorus ludovicianus"), ("barn_swallow", "Hirundo rustica"),
    ("stellers_jay", "Cyanocitta stelleri"), ("common_raven", "Corvus corax"),
    ("atlantic_puffin", "Fratercula arctica"), ("american_flamingo", "Phoenicopterus ruber"),
    ("gray_squirrel", "Sciurus carolinensis"), ("white_tailed_deer", "Odocoileus virginianus"),
    ("domestic_cat", "Felis catus"), ("red_fox", "Vulpes vulpes"),
    ("monarch_butterfly", "Danaus plexippus"), ("sunflower", "Helianthus annuus"),
]
NON_BIRDS = {"gray_squirrel", "white_tailed_deer", "domestic_cat", "red_fox", "monarch_butterfly", "sunflower"}


def fetch(url: str, accept: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download() -> None:
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    for tag, name in OTHER_TAXA:
        have = len(list(OOD_DIR.glob(f"{tag}_*.jpg")))
        if have >= PER_TAXON:
            continue
        query = urllib.parse.urlencode(
            {"taxon_name": name, "quality_grade": "research", "photos": "true", "per_page": 20,
             "order_by": "id", "order": "desc"}
        )
        try:
            data = json.loads(fetch(f"{INAT_API}/observations?{query}", "application/json"))
        except Exception as exc:
            print(f"skip {tag}: {exc}", flush=True)
            continue
        saved = have
        for observation in data["results"]:
            if saved >= PER_TAXON:
                break
            if not observation.get("photos"):
                continue
            url = observation["photos"][0]["url"].replace("square", "large")
            try:
                (OOD_DIR / f"{tag}_{saved:02d}.jpg").write_bytes(fetch(url, "image/*"))
                saved += 1
            except Exception as exc:
                print(f"skip photo {tag}: {exc}", flush=True)
            time.sleep(1)
        print(f"{tag} {saved}", flush=True)
        time.sleep(1)


def pooled_features(net: torch.nn.Module, paths: list[str]) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch = torch.stack([image_tensor_from_path(Path(path)) for path in paths[start : start + 32]])
            out.append(torch.flatten(net.avgpool(net.features(batch)), 1).numpy())
    return np.concatenate(out)


def gate_scores(features: np.ndarray, weights: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((features - mean) / scale) @ weights + bias


def fit() -> None:
    from sklearn.linear_model import LogisticRegression

    import dashboard

    dashboard.load_models()
    net = dashboard.IMAGE_NET
    train_rows, calib_rows, val_rows = (read_split("image", name) for name in ("train", "calib", "val"))
    known = {
        name: pooled_features(net, [row["file_path"] for row in rows])
        for name, rows in (("train", train_rows), ("calib", calib_rows), ("val", val_rows))
    }
    files = sorted(OOD_DIR.glob("*.jpg"))
    if not files:
        raise SystemExit("no photos in data/ood; run python photo_gate.py download")
    tags = np.array([path.stem.rsplit("_", 1)[0] for path in files])
    other = pooled_features(net, [str(path) for path in files])

    mean = known["train"].mean(axis=0)
    scale = known["train"].std(axis=0) + 1e-6

    def train_gate(other_features: np.ndarray):
        x = np.r_[known["train"], other_features]
        y = np.r_[np.ones(len(known["train"])), np.zeros(len(other_features))]
        model = LogisticRegression(C=0.02, max_iter=3000, class_weight="balanced")
        model.fit((x - mean) / scale, y)
        return model.coef_[0], float(model.intercept_[0])

    # Honest estimate: hold each unlisted taxon out, fit on the rest, then see how often it is caught.
    caught_by_tag: dict[str, float] = {}
    kept_val: list[float] = []
    for tag in sorted(set(tags)):
        weights, bias = train_gate(other[tags != tag])
        cut = np.quantile(gate_scores(known["calib"], weights, bias, mean, scale), 1 - KEEP)
        caught_by_tag[tag] = float((gate_scores(other[tags == tag], weights, bias, mean, scale) < cut).mean())
        kept_val.append(float((gate_scores(known["val"], weights, bias, mean, scale) >= cut).mean()))
    birds = [tag for tag in caught_by_tag if tag not in NON_BIRDS]

    weights, bias = train_gate(other)
    cut = float(np.quantile(gate_scores(known["calib"], weights, bias, mean, scale), 1 - KEEP))
    np.savez(GATE_PATH, weights=weights.astype(np.float32), bias=np.float32(bias),
             mean=mean.astype(np.float32), scale=scale.astype(np.float32), cut=np.float32(cut))
    report = {
        "what": "logistic regression on pooled photo features: the 20 species vs photos of other species",
        "keep_of_known_photos": KEEP,
        "other_taxa": len(set(tags)),
        "other_photos": int(len(tags)),
        "unlisted_birds_caught_when_held_out": round(float(np.mean([caught_by_tag[t] for t in birds])), 3),
        "non_birds_caught_when_held_out": round(float(np.mean([caught_by_tag[t] for t in caught_by_tag if t in NON_BIRDS] or [0])), 3),
        "val_known_photos_kept": round(float(np.mean(kept_val)), 3),
        "caught_by_taxon": {tag: round(value, 2) for tag, value in sorted(caught_by_tag.items())},
        "note": "Held-out taxa were never in the fit that scored them. Test photos were not read.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "caught_by_taxon"}, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
    elif len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    else:
        raise SystemExit(__doc__)
