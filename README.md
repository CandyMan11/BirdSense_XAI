The titles were sitting inside a code block, so GitHub showed the `#` marks as plain text. Paste the text below into the README editor with no code fence around it. The `#` lines become the large titles.

# BirdSense-XAI

BirdSense-XAI classifies 20 bird species from a photo, a recording, or both. A photo model and a recording model are trained separately. A fusion layer then combines their scores. A local dashboard runs a single pair and shows which source the fused call followed.

Repository: https://github.com/CandyMan11/BirdSense_XAI

The photos and recordings stay on your machine. Git tracks the code and the saved checkpoints.

## Prerequisites

- Python 3.10 or newer
- About 4 GB of free disk space for the media and the mel cache
- Network access to [iNaturalist](https://www.inaturalist.org/) while downloading
- Network access on the first training run, so torchvision can fetch the MobileNet V3 Small weights

The download script uses the Python standard library. Training, fusion, and the dashboard need these packages:

```text
torch
torchvision
numpy
Pillow
librosa
soundfile
av
```

The scripts run on CPU.

## Setup

From this folder, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision numpy Pillow librosa soundfile av
```

## Download the dataset

```powershell
python download_dataset.py
```

This collects a closed set of 20 species. A species is kept only when it has at least 80 photos and 80 recordings. The target is 100 files per modality. One file is taken from each research-grade observation, walking records in ascending observation id, so a rerun stays stable. Photos are the iNaturalist large size. API calls are spaced by about 1 second. Files already on disk are skipped.

Media comes from iNaturalist research-grade observations through https://api.inaturalist.org/v1. Each file's observation page, media URL, license, and attribution are written to `data/metadata/manifest.csv`. License and attribution stay with the iNaturalist observer named there.

Set `XENOCANTO_API_KEY` before the download only if you want [Xeno-canto](https://xeno-canto.org/) to fill in a species that has fewer than 100 iNaturalist recordings. The current set did not use Xeno-canto.

The download also writes `data/metadata/dataset_summary.md`. It does not change this README.

Splits cover kept files only: train 60%, val 15%, calib 10%, and test 15%, stratified by species and modality, seed 42. Current kept sizes are train 2400, val 600, calib 400, and test 600.

| Common name | Scientific name | Photos | Recordings |
| --- | --- | ---: | ---: |
| Canada Goose | Branta canadensis | 100 | 100 |
| Wood Duck | Aix sponsa | 100 | 100 |
| Mallard | Anas platyrhynchos | 100 | 100 |
| Mourning Dove | Zenaida macroura | 100 | 100 |
| Downy Woodpecker | Dryobates pubescens | 100 | 100 |
| Red-bellied Woodpecker | Melanerpes carolinus | 100 | 100 |
| Blue Jay | Cyanocitta cristata | 100 | 100 |
| American Crow | Corvus brachyrhynchos | 100 | 100 |
| Black-capped Chickadee | Poecile atricapillus | 100 | 100 |
| Tufted Titmouse | Baeolophus bicolor | 100 | 100 |
| White-breasted Nuthatch | Sitta carolinensis | 100 | 100 |
| American Robin | Turdus migratorius | 100 | 100 |
| Gray Catbird | Dumetella carolinensis | 100 | 100 |
| Song Sparrow | Melospiza melodia | 100 | 100 |
| Northern Cardinal | Cardinalis cardinalis | 100 | 100 |
| Red-winged Blackbird | Agelaius phoeniceus | 100 | 100 |
| Common Grackle | Quiscalus quiscula | 100 | 100 |
| Baltimore Oriole | Icterus galbula | 100 | 100 |
| American Goldfinch | Spinus tristis | 100 | 100 |
| House Sparrow | Passer domesticus | 100 | 100 |

## Train

Checkpoints are already in `models/`. Run training only to rebuild them. Weights update on the train split. The val split chooses the checkpoint. Calib and test are not read.

```powershell
python train.py
python train.py image
python train.py audio
python train.py improve
```

`python train.py` trains both modalities. `improve` fine-tunes the photo network from `models/image_head_best.pt` and replaces the recording network with a MobileNet trained on log-mel images.

Photos use MobileNet V3 Small. The backbone starts frozen, and `improve` unfreezes the last four feature blocks. Recordings are decoded to mono at 22,050 Hz. Each training clip is 3 seconds, converted to a 64-band log-mel and resized to 224×224.

## Fuse

Photos and recordings come from different observations, so a pair is one photo and one recording of the same species from the same split. The fusion rule is chosen on val. Temperature is fit on calib. The test split is scored once.

```powershell
python fuse.py
python fuse.py windows
```

`windows` scores each recording as the mean of up to eight 3-second windows, capped at 120 seconds of audio. It keeps that pooling only when validation accuracy beats the saved first-window result. Results land in `data/metadata/fusion_results.csv` and `data/metadata/fusion_test_by_species.csv`.

## Dashboard

The dashboard serves inference on this machine. It does not train, and it does not rescore the test split.

```powershell
python dashboard.py
```

Open http://127.0.0.1:8765. It needs the downloaded files plus `models/image_best.pt`, `models/audio_best.pt`, and `models/fusion_best.pt`. Pick a photo and a recording. The page shows the fused species, the photo-only call, the recording-only call, and how much each modality contributed to the fused logit.

## Layout

| Path | Role |
| --- | --- |
| `download_dataset.py` | Downloads media and writes the splits |
| `train.py` | Trains the photo and recording models |
| `fuse.py` | Chooses and scores the fusion rule |
| `dashboard.py` | Local inference server |
| `dashboard/index.html` | Dashboard page |
| `models/` | Saved checkpoints |
| `data/raw/` | Local photos and recordings |
| `data/metadata/` | Manifest, splits, and result tables |
| `data/cache/` | Mel spectrograms and cached logits |

`data/` is listed in `.gitignore`.
