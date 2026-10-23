"""Download the BirdSense-XAI closed species set and record what was kept.

This step only collects data. It does not train a model.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "data" / "raw" / "images"
AUDIO_ROOT = ROOT / "data" / "raw" / "audio"
METADATA = ROOT / "data" / "metadata"
MANIFEST_PATH = METADATA / "manifest.csv"
FILTER_PATH = METADATA / "species_filter_log.csv"
SPLITS_PATH = METADATA / "splits.csv"
README_PATH = ROOT / "README.md"
LOG_PATH = METADATA / "download_log.txt"
STATUS_PATH = METADATA / "status.txt"

TARGET = 100
MIN_KEEP = 80
PER_PAGE = 100
MAX_PAGES = 30
SPLIT_SEED = 42
API_INTERVAL_SECONDS = 1.0
API_TIMEOUT_SECONDS = 120
MEDIA_TIMEOUT_SECONDS = 300
MIN_BYTES = 1024
USER_AGENT = "BirdSense-XAI/0.1 (educational bird dataset; research-grade media download)"

INAT_API = "https://api.inaturalist.org/v1"
XC_API = "https://xeno-canto.org/api/3/recordings"

SPECIES = [
    ("Canada Goose", "Branta canadensis"),
    ("Wood Duck", "Aix sponsa"),
    ("Mallard", "Anas platyrhynchos"),
    ("Mourning Dove", "Zenaida macroura"),
    ("Downy Woodpecker", "Dryobates pubescens"),
    ("Red-bellied Woodpecker", "Melanerpes carolinus"),
    ("Blue Jay", "Cyanocitta cristata"),
    ("American Crow", "Corvus brachyrhynchos"),
    ("Black-capped Chickadee", "Poecile atricapillus"),
    ("Tufted Titmouse", "Baeolophus bicolor"),
    ("White-breasted Nuthatch", "Sitta carolinensis"),
    ("American Robin", "Turdus migratorius"),
    ("Gray Catbird", "Dumetella carolinensis"),
    ("Song Sparrow", "Melospiza melodia"),
    ("Northern Cardinal", "Cardinalis cardinalis"),
    ("Red-winged Blackbird", "Agelaius phoeniceus"),
    ("Common Grackle", "Quiscalus quiscula"),
    ("Baltimore Oriole", "Icterus galbula"),
    ("American Goldfinch", "Spinus tristis"),
    ("House Sparrow", "Passer domesticus"),
]

SPLIT_FRACTIONS = (
    ("train", 0.60),
    ("val", 0.15),
    ("calib", 0.10),
    ("test", 0.15),
)

MANIFEST_FIELDS = [
    "sample_id",
    "common_name",
    "scientific_name",
    "taxon_id",
    "modality",
    "source",
    "source_record_id",
    "media_id",
    "file_path",
    "license",
    "attribution",
    "observed_on",
    "latitude",
    "longitude",
    "geoprivacy",
    "quality_grade",
    "source_url",
    "media_url",
    "kept",
]

FILTER_FIELDS = [
    "species_order",
    "common_name",
    "scientific_name",
    "taxon_id",
    "n_image",
    "n_audio",
    "target_per_modality",
    "minimum_to_keep",
    "kept",
    "image_source",
    "audio_source",
    "note",
]

SPLIT_FIELDS = [
    "sample_id",
    "file_path",
    "common_name",
    "scientific_name",
    "modality",
    "source",
    "split",
]

_LAST_API_AT = None


def self_check() -> None:
    """Fail fast if the split arithmetic drifts from the requested ratios."""
    exact = split_counts(100)
    if exact != {"train": 60, "val": 15, "calib": 10, "test": 15}:
        raise RuntimeError(f"split_counts(100) returned {exact}")
    eighty = split_counts(80)
    if eighty != {"train": 48, "val": 12, "calib": 8, "test": 12}:
        raise RuntimeError(f"split_counts(80) returned {eighty}")
    for count in range(0, 121):
        parts = split_counts(count)
        if sum(parts.values()) != count:
            raise RuntimeError(f"split_counts({count}) does not sum to n")
        if set(parts) != {"train", "val", "calib", "test"}:
            raise RuntimeError(f"split_counts({count}) has unexpected keys")


def split_counts(n: int) -> dict[str, int]:
    """Largest-remainder allocation of train 60%, val 15%, calib 10%, test 15%."""
    raw = {name: n * fraction for name, fraction in SPLIT_FRACTIONS}
    counts = {name: int(raw[name]) for name, _ in SPLIT_FRACTIONS}
    leftover = n - sum(counts.values())
    ranked = sorted(
        SPLIT_FRACTIONS,
        key=lambda item: (raw[item[0]] - counts[item[0]], -SPLIT_FRACTIONS.index(item)),
        reverse=True,
    )
    for name, _ in ranked[:leftover]:
        counts[name] += 1
    return counts


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def set_status(message: str) -> None:
    METADATA.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(f"pid={os.getpid()}\n{message}\n", encoding="utf-8")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def slug(scientific_name: str) -> str:
    return scientific_name.strip().lower().replace(" ", "_")


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pace_api() -> None:
    """Keep public API calls about one second apart."""
    global _LAST_API_AT
    now = time.monotonic()
    if _LAST_API_AT is not None:
        delay = API_INTERVAL_SECONDS - (now - _LAST_API_AT)
        if delay > 0:
            time.sleep(delay)
    _LAST_API_AT = time.monotonic()


def api_get(url: str, label: str) -> dict:
    last_error = "unknown error"
    for attempt in range(1, 5):
        pace_api()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("API response was not a JSON object")
            return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read(200).decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code} {detail}"
            if exc.code == 429 or exc.code >= 500:
                retry_after = exc.headers.get("Retry-After", "5")
                try:
                    extra = max(float(retry_after), 1.0)
                except ValueError:
                    extra = 5.0
                log(f"retry {attempt}/4 {label}: {last_error}")
                time.sleep(extra)
                continue
            raise RuntimeError(f"{label} failed: {last_error}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = clean_text(exc)[:200]
            log(f"retry {attempt}/4 {label}: {last_error}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"{label} failed after retries: {last_error}")


def xeno_canto_key() -> str:
    return os.environ.get("XENOCANTO_API_KEY", "").strip()


def resolve_taxon(scientific_name: str) -> int:
    url = INAT_API + "/taxa?" + urllib.parse.urlencode(
        {
            "q": scientific_name,
            "rank": "species",
            "is_active": "true",
            "per_page": "20",
        }
    )
    payload = api_get(url, f"taxon {scientific_name}")
    for taxon in payload.get("results") or []:
        name = clean_text(taxon.get("name")).lower()
        if (
            name == scientific_name.lower()
            and taxon.get("rank") == "species"
            and taxon.get("is_active")
        ):
            return int(taxon["id"])
    raise RuntimeError(f"no active species taxon for {scientific_name}")


def iter_observations(taxon_id: int, media: str, scientific_name: str):
    """Walk research-grade observations that have the requested media.

    photos=false and sounds=false are real exclusion filters on this API,
    so the unused flag is omitted rather than set false.
    """
    if media not in {"photos", "sounds"}:
        raise ValueError(media)
    params = {
        "taxon_id": str(taxon_id),
        "quality_grade": "research",
        "per_page": str(PER_PAGE),
        "order_by": "id",
        "order": "asc",
        media: "true",
    }
    seen_ids: set[str] = set()
    for page in range(1, MAX_PAGES + 1):
        params["page"] = str(page)
        url = INAT_API + "/observations?" + urllib.parse.urlencode(params)
        label = f"observations {scientific_name} {media} page {page}"
        try:
            payload = api_get(url, label)
        except Exception as exc:
            log(f"stopping {scientific_name} {media} after page error: {clean_text(exc)[:180]}")
            break
        results = payload.get("results") or []
        total = payload.get("total_results")
        log(f"{label}: received {len(results)} of about {total}")
        if not results:
            break
        for observation in results:
            observation_id = str(observation.get("id") or "")
            if not observation_id or observation_id in seen_ids:
                continue
            seen_ids.add(observation_id)
            yield observation
        if len(results) < PER_PAGE:
            break


def parse_latlon(observation: dict) -> tuple[str, str]:
    location = clean_text(observation.get("location"))
    if "," not in location:
        return "", ""
    latitude, longitude = location.split(",", 1)
    return latitude.strip(), longitude.strip()


def extension_from_url(url: str, default: str) -> str:
    path = urllib.parse.urlsplit(url).path.lower()
    suffix = Path(path).suffix
    allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4"}
    if suffix in allowed:
        return suffix
    return default


def resized_photo_url(url: str, size: str) -> str:
    parts = urllib.parse.urlsplit(url)
    name = parts.path.rsplit("/", 1)[-1]
    if "." not in name:
        return url
    label, ext = name.split(".", 1)
    if label not in {"square", "thumb", "small", "medium", "large", "original"}:
        return url
    new_path = parts.path[: -len(name)] + f"{size}.{ext}"
    return urllib.parse.urlunsplit(parts._replace(path=new_path))


def photo_candidates(photo: dict) -> list[str]:
    url = clean_text(photo.get("url"))
    if not url:
        return []
    ordered = [
        resized_photo_url(url, "large"),
        resized_photo_url(url, "original"),
        resized_photo_url(url, "medium"),
    ]
    unique = []
    for candidate in ordered:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def file_header(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(16)


def looks_like_image(path: Path) -> bool:
    header = file_header(path)
    if header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG") or header.startswith(b"GIF8"):
        return True
    return header.startswith(b"RIFF") and header[8:12] == b"WEBP"


def looks_like_audio(path: Path) -> bool:
    header = file_header(path)
    if header.startswith((b"ID3", b"OggS", b"fLaC", b"RIFF")):
        return True
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    return len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0


def file_is_valid(path: Path, modality: str) -> bool:
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False
    return looks_like_image(path) if modality == "image" else looks_like_audio(path)


def download_file(url: str, dest: Path, modality: str) -> str:
    """Download url to dest. Return 'skipped' or 'downloaded'."""
    if file_is_valid(dest, modality):
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=MEDIA_TIMEOUT_SECONDS) as response:
            content_type = clean_text(response.headers.get("Content-Type")).lower()
            if content_type.startswith("text/") or "html" in content_type:
                raise RuntimeError(f"unexpected content type {content_type}")
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if not file_is_valid(temporary, modality):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("downloaded file failed validation")
    if dest.exists():
        dest.unlink()
    temporary.replace(dest)
    return "downloaded"


def save_with_retries(urls: list[str], dest: Path, modality: str) -> str | None:
    """Try each URL a few times. Return the URL that was saved, or None."""
    for url in urls:
        for attempt in range(1, 4):
            try:
                download_file(url, dest, modality)
                return url
            except Exception as exc:
                if file_is_valid(dest, modality):
                    return url
                log(f"media retry {attempt}/3 {dest.name}: {clean_text(exc)[:180]}")
                time.sleep(attempt)
    return None


def base_row(
    common_name: str,
    scientific_name: str,
    taxon_id: int,
    modality: str,
    source: str,
    source_record_id: str,
    media_id: str,
    file_path: str,
    license_code: str,
    attribution: str,
    observed_on: str,
    latitude: str,
    longitude: str,
    geoprivacy: str,
    quality_grade: str,
    source_url: str,
    media_url: str,
) -> dict[str, str]:
    return {
        "sample_id": f"{source}_{modality}_{source_record_id}_{media_id}",
        "common_name": common_name,
        "scientific_name": scientific_name,
        "taxon_id": str(taxon_id),
        "modality": modality,
        "source": source,
        "source_record_id": source_record_id,
        "media_id": media_id,
        "file_path": file_path,
        "license": clean_text(license_code),
        "attribution": clean_text(attribution),
        "observed_on": clean_text(observed_on),
        "latitude": latitude,
        "longitude": longitude,
        "geoprivacy": clean_text(geoprivacy),
        "quality_grade": clean_text(quality_grade),
        "source_url": source_url,
        "media_url": media_url,
        "kept": "",
    }


def collect_inaturalist_images(
    common_name: str,
    scientific_name: str,
    taxon_id: int,
) -> list[dict[str, str]]:
    rows = []
    species_dir = IMAGE_ROOT / slug(scientific_name)
    for observation in iter_observations(taxon_id, "photos", scientific_name):
        if len(rows) >= TARGET:
            break
        if observation.get("quality_grade") != "research":
            continue
        observation_id = str(observation.get("id") or "")
        latitude, longitude = parse_latlon(observation)
        geoprivacy = clean_text(observation.get("geoprivacy") or observation.get("taxon_geoprivacy"))
        source_url = clean_text(observation.get("uri")) or f"https://www.inaturalist.org/observations/{observation_id}"
        for photo in observation.get("photos") or []:
            if photo.get("hidden") or not photo.get("id"):
                continue
            media_id = str(photo["id"])
            candidates = photo_candidates(photo)
            if not candidates:
                continue
            extension = extension_from_url(candidates[0], ".jpg")
            dest = species_dir / f"{observation_id}_{media_id}{extension}"
            saved_url = save_with_retries(candidates, dest, "image")
            if not saved_url:
                continue
            rows.append(
                base_row(
                    common_name,
                    scientific_name,
                    taxon_id,
                    "image",
                    "inaturalist",
                    observation_id,
                    media_id,
                    relative_path(dest),
                    photo.get("license_code") or "",
                    photo.get("attribution") or "",
                    observation.get("observed_on") or "",
                    latitude,
                    longitude,
                    geoprivacy,
                    observation.get("quality_grade") or "",
                    source_url,
                    saved_url,
                )
            )
            log(f"image {len(rows)}/{TARGET} {scientific_name} {dest.name}")
            set_status(f"image {len(rows)}/{TARGET} {common_name}")
            break
    return rows


def collect_inaturalist_audio(
    common_name: str,
    scientific_name: str,
    taxon_id: int,
) -> list[dict[str, str]]:
    rows = []
    species_dir = AUDIO_ROOT / slug(scientific_name)
    for observation in iter_observations(taxon_id, "sounds", scientific_name):
        if len(rows) >= TARGET:
            break
        if observation.get("quality_grade") != "research":
            continue
        observation_id = str(observation.get("id") or "")
        latitude, longitude = parse_latlon(observation)
        geoprivacy = clean_text(observation.get("geoprivacy") or observation.get("taxon_geoprivacy"))
        source_url = clean_text(observation.get("uri")) or f"https://www.inaturalist.org/observations/{observation_id}"
        for sound in observation.get("sounds") or []:
            if sound.get("hidden") or not sound.get("file_url") or not sound.get("id"):
                continue
            media_id = str(sound["id"])
            media_url = clean_text(sound.get("file_url"))
            extension = extension_from_url(media_url, ".mp3")
            dest = species_dir / f"{observation_id}_{media_id}{extension}"
            saved_url = save_with_retries([media_url], dest, "audio")
            if not saved_url:
                continue
            rows.append(
                base_row(
                    common_name,
                    scientific_name,
                    taxon_id,
                    "audio",
                    "inaturalist",
                    observation_id,
                    media_id,
                    relative_path(dest),
                    sound.get("license_code") or "",
                    sound.get("attribution") or "",
                    observation.get("observed_on") or "",
                    latitude,
                    longitude,
                    geoprivacy,
                    observation.get("quality_grade") or "",
                    source_url,
                    saved_url,
                )
            )
            log(f"audio {len(rows)}/{TARGET} {scientific_name} {dest.name}")
            set_status(f"audio {len(rows)}/{TARGET} {common_name}")
            break
    return rows


def collect_xeno_canto_audio(
    common_name: str,
    scientific_name: str,
    taxon_id: int,
    already: int,
) -> list[dict[str, str]]:
    key = xeno_canto_key()
    if not key or already >= TARGET:
        return []
    rows = []
    species_dir = AUDIO_ROOT / slug(scientific_name)
    needed = TARGET - already
    for quality in ("A", "B"):
        if len(rows) >= needed:
            break
        for page in range(1, 6):
            if len(rows) >= needed:
                break
            query = f"{scientific_name} q:{quality}"
            url = XC_API + "?" + urllib.parse.urlencode(
                {"query": query, "page": str(page), "key": key}
            )
            payload = api_get(url, f"xeno-canto {scientific_name} q:{quality} page {page}")
            recordings = payload.get("recordings") or []
            log(f"xeno-canto {scientific_name} q:{quality} page {page}: {len(recordings)} recordings")
            if not recordings:
                break
            for recording in recordings:
                if len(rows) >= needed:
                    break
                genus = clean_text(recording.get("gen"))
                species = clean_text(recording.get("sp"))
                if f"{genus} {species}".lower() != scientific_name.lower():
                    continue
                if clean_text(recording.get("q")).upper() != quality:
                    continue
                record_id = clean_text(recording.get("id"))
                file_url = clean_text(recording.get("file"))
                if not record_id or not file_url:
                    continue
                extension = extension_from_url(file_url, ".mp3")
                dest = species_dir / f"xc_{record_id}{extension}"
                saved_url = save_with_retries([file_url], dest, "audio")
                if not saved_url:
                    continue
                public_file_url = urllib.parse.urlsplit(saved_url)
                query = urllib.parse.parse_qsl(public_file_url.query, keep_blank_values=True)
                query = [(name, value) for name, value in query if name.lower() != "key"]
                safe_url = urllib.parse.urlunsplit(
                    public_file_url._replace(query=urllib.parse.urlencode(query))
                )
                rows.append(
                    base_row(
                        common_name,
                        scientific_name,
                        taxon_id,
                        "audio",
                        "xeno-canto",
                        record_id,
                        record_id,
                        relative_path(dest),
                        recording.get("lic") or "",
                        f"Recorded by {clean_text(recording.get('rec'))}; Xeno-canto XC{record_id}",
                        recording.get("date") or "",
                        clean_text(recording.get("lat")),
                        clean_text(recording.get("lng")),
                        "",
                        f"xeno-canto q:{quality}",
                        clean_text(recording.get("url")) or f"https://xeno-canto.org/{record_id}",
                        safe_url,
                    )
                )
                log(f"audio xeno-canto {already + len(rows)}/{TARGET} {scientific_name} {dest.name}")
            pages = int(payload.get("numPages") or 1)
            if page >= pages:
                break
    return rows


def source_label(rows: list[dict[str, str]], modality: str) -> str:
    sources = sorted({row["source"] for row in rows if row["modality"] == modality})
    if not sources:
        return "none"
    if sources == ["inaturalist"]:
        return "inaturalist"
    if sources == ["xeno-canto"]:
        return "xeno-canto"
    return "inaturalist+xeno-canto"


def species_note(n_image: int, n_audio: int, kept: bool) -> str:
    gaps = []
    if n_image < MIN_KEEP:
        gaps.append(f"images {n_image} < {MIN_KEEP}")
    if n_audio < MIN_KEEP:
        gaps.append(f"audio {n_audio} < {MIN_KEEP}")
    if gaps:
        text = "not kept: " + "; ".join(gaps)
    elif kept:
        text = "kept"
    else:
        text = "not kept"
    if n_image < TARGET or n_audio < TARGET:
        text += f"; short of target {TARGET} (images {n_image}, audio {n_audio})"
    return text


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def assign_splits(kept_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rng = random.Random(SPLIT_SEED)
    split_rows = []
    for common_name, scientific_name in SPECIES:
        for modality in ("image", "audio"):
            group = [
                row
                for row in kept_rows
                if row["scientific_name"] == scientific_name and row["modality"] == modality
            ]
            if not group:
                continue
            rng.shuffle(group)
            counts = split_counts(len(group))
            offset = 0
            for split_name, _ in SPLIT_FRACTIONS:
                take = counts[split_name]
                for row in group[offset : offset + take]:
                    split_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "file_path": row["file_path"],
                            "common_name": common_name,
                            "scientific_name": scientific_name,
                            "modality": modality,
                            "source": row["source"],
                            "split": split_name,
                        }
                    )
                offset += take
            if offset != len(group):
                raise RuntimeError(f"split assignment missed rows for {scientific_name} {modality}")
    return split_rows


def write_readme(filter_rows: list[dict[str, str]], split_rows: list[dict[str, str]]) -> None:
    lines = [
        "# BirdSense-XAI dataset",
        "",
        "Repository: https://github.com/CandyMan11/BirdSense_XAI",
        "",
        "Closed 20-species download of research-grade iNaturalist photos and recordings.",
        "A species is kept only when it has at least 80 photos and 80 recordings.",
        "The download target is 100 files per modality. One file is taken from each observation,",
        "walking research-grade records in ascending observation id so a rerun stays stable.",
        "Photos are the iNaturalist large size (longest side 1024 pixels).",
        "API calls are spaced by about 1 second. Files already on disk are skipped.",
        "",
    ]
    if xeno_canto_key():
        lines.append(
            "Xeno-canto is used only to top up a species that has fewer than 100 iNaturalist recordings."
        )
    else:
        lines.append("Xeno-canto was not used because XENOCANTO_API_KEY was not set.")
    lines.extend(
        [
            "",
            "Photos and recordings were acquired from iNaturalist research-grade observations",
            "(https://www.inaturalist.org/) through https://api.inaturalist.org/v1.",
            "Each saved file is one observation. The observation page, media URL, license,",
            "and attribution are recorded in data/metadata/manifest.csv.",
            "License and attribution stay with the iNaturalist observer named there.",
            "The files under data/ stay on disk and are not part of the git repository.",
            "",
            "The manifest lists every saved file. Splits include kept species only:",
            "train 60%, val 15%, calib 10%, and test 15%, stratified by species and modality",
            f"(seed {SPLIT_SEED}). This step does not train a model.",
            "",
            "## Species counts",
            "",
            "| Common name | Scientific name | n_image | n_audio | Kept | Audio source |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    image_total = 0
    audio_total = 0
    for row in filter_rows:
        image_total += int(row["n_image"])
        audio_total += int(row["n_audio"])
        lines.append(
            f"| {row['common_name']} | {row['scientific_name']} | {row['n_image']} | "
            f"{row['n_audio']} | {row['kept']} | {row['audio_source']} |"
        )
    lines.append(f"| Total |  | {image_total} | {audio_total} |  |  |")
    split_totals: dict[str, int] = {}
    for row in split_rows:
        split_totals[row["split"]] = split_totals.get(row["split"], 0) + 1
    summary = ", ".join(f"{name} {split_totals.get(name, 0)}" for name, _ in SPLIT_FRACTIONS)
    lines.extend(
        [
            "",
            f"Kept split sizes: {summary}.",
            "",
            "Rerun from this folder with `python download_dataset.py`.",
            "",
        ]
    )
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(records: list[dict[str, str]], filter_rows: list[dict[str, str]]) -> None:
    sample_ids = [row["sample_id"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("manifest sample_id values are not unique")
    for row in records:
        path = ROOT / row["file_path"]
        if not file_is_valid(path, row["modality"]):
            raise RuntimeError(f"manifest points at a missing file: {row['file_path']}")
    kept_rows = [row for row in records if row["kept"] == "yes"]
    split_rows = assign_splits(kept_rows)
    if {row["sample_id"] for row in split_rows} != {row["sample_id"] for row in kept_rows}:
        raise RuntimeError("splits do not match the kept manifest rows")
    write_csv(MANIFEST_PATH, MANIFEST_FIELDS, records)
    write_csv(FILTER_PATH, FILTER_FIELDS, filter_rows)
    write_csv(SPLITS_PATH, SPLIT_FIELDS, split_rows)
    write_readme(filter_rows, split_rows)


def filter_row(
    order: int,
    common_name: str,
    scientific_name: str,
    taxon_id: str,
    n_image: int,
    n_audio: int,
    kept: str,
    image_source: str,
    audio_source: str,
    note: str,
) -> dict[str, str]:
    return {
        "species_order": str(order),
        "common_name": common_name,
        "scientific_name": scientific_name,
        "taxon_id": taxon_id,
        "n_image": str(n_image),
        "n_audio": str(n_audio),
        "target_per_modality": str(TARGET),
        "minimum_to_keep": str(MIN_KEEP),
        "kept": kept,
        "image_source": image_source,
        "audio_source": audio_source,
        "note": note,
    }


def process_species(
    order: int,
    common_name: str,
    scientific_name: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    log(f"START {order}/20 {common_name} ({scientific_name})")
    set_status(f"START {order}/20 {common_name}")
    try:
        taxon_id = resolve_taxon(scientific_name)
    except Exception as exc:
        note = f"taxon lookup failed: {clean_text(exc)[:180]}"
        log(f"UNDER_MINIMUM {common_name} n_image=0 n_audio=0 {note}")
        return (
            filter_row(order, common_name, scientific_name, "", 0, 0, "no", "none", "none", note),
            [],
        )

    images = collect_inaturalist_images(common_name, scientific_name, taxon_id)
    audio = collect_inaturalist_audio(common_name, scientific_name, taxon_id)
    if len(audio) < TARGET and xeno_canto_key():
        audio.extend(collect_xeno_canto_audio(common_name, scientific_name, taxon_id, len(audio)))
    elif len(audio) < TARGET:
        log(f"{scientific_name} audio {len(audio)}/{TARGET}; Xeno-canto key is not set")

    rows = images + audio
    n_image = len(images)
    n_audio = len(audio)
    kept_flag = n_image >= MIN_KEEP and n_audio >= MIN_KEEP
    kept = "yes" if kept_flag else "no"
    for row in rows:
        row["kept"] = kept
    image_source = source_label(rows, "image")
    audio_source = source_label(rows, "audio")
    note = species_note(n_image, n_audio, kept_flag)
    summary = (
        f"SUMMARY {common_name}|{scientific_name}|n_image={n_image}|n_audio={n_audio}|"
        f"kept={kept}|audio_source={audio_source}"
    )
    log(summary)
    if not kept_flag:
        log(f"UNDER_MINIMUM {common_name} n_image={n_image} n_audio={n_audio}")
    return (
        filter_row(
            order,
            common_name,
            scientific_name,
            str(taxon_id),
            n_image,
            n_audio,
            kept,
            image_source,
            audio_source,
            note,
        ),
        rows,
    )


def main() -> None:
    self_check()
    METADATA.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    if xeno_canto_key():
        log("XENOCANTO_API_KEY is set; it will top up audio only when iNaturalist has fewer than 100 recordings")
    else:
        log("XENOCANTO_API_KEY is not set; audio will come from iNaturalist only")

    records: list[dict[str, str]] = []
    filter_rows: list[dict[str, str]] = []
    try:
        for order, (common_name, scientific_name) in enumerate(SPECIES, start=1):
            species_log, species_rows = process_species(order, common_name, scientific_name)
            filter_rows.append(species_log)
            records.extend(species_rows)
            write_outputs(records, filter_rows)
        log("DONE")
        set_status("DONE")
    except KeyboardInterrupt:
        log("INTERRUPTED")
        write_outputs(records, filter_rows)
        raise
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {clean_text(exc)[:300]}")
        if filter_rows:
            write_outputs(records, filter_rows)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
