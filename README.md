# BirdSense-XAI dataset

Repository: https://github.com/CandyMan11/BirdSense_XAI

Closed 20-species download of research-grade iNaturalist photos and recordings.
A species is kept only when it has at least 80 photos and 80 recordings.
The download target is 100 files per modality. One file is taken from each observation,
walking research-grade records in ascending observation id so a rerun stays stable.
Photos are the iNaturalist large size (longest side 1024 pixels).
API calls are spaced by about 1 second. Files already on disk are skipped.

Xeno-canto was not used because XENOCANTO_API_KEY was not set.

Photos and recordings were acquired from iNaturalist research-grade observations
(https://www.inaturalist.org/) through https://api.inaturalist.org/v1.
Each saved file is one observation. The observation page, media URL, license,
and attribution are recorded in data/metadata/manifest.csv.
License and attribution stay with the iNaturalist observer named there.
The files under data/ stay on disk and are not part of the git repository.

The manifest lists every saved file. Splits include kept species only:
train 60%, val 15%, calib 10%, and test 15%, stratified by species and modality
(seed 42). This step does not train a model.

## Species counts

| Common name | Scientific name | n_image | n_audio | Kept | Audio source |
| --- | --- | ---: | ---: | --- | --- |
| Canada Goose | Branta canadensis | 100 | 100 | yes | inaturalist |
| Wood Duck | Aix sponsa | 100 | 100 | yes | inaturalist |
| Mallard | Anas platyrhynchos | 100 | 100 | yes | inaturalist |
| Mourning Dove | Zenaida macroura | 100 | 100 | yes | inaturalist |
| Downy Woodpecker | Dryobates pubescens | 100 | 100 | yes | inaturalist |
| Red-bellied Woodpecker | Melanerpes carolinus | 100 | 100 | yes | inaturalist |
| Blue Jay | Cyanocitta cristata | 100 | 100 | yes | inaturalist |
| American Crow | Corvus brachyrhynchos | 100 | 100 | yes | inaturalist |
| Black-capped Chickadee | Poecile atricapillus | 100 | 100 | yes | inaturalist |
| Tufted Titmouse | Baeolophus bicolor | 100 | 100 | yes | inaturalist |
| White-breasted Nuthatch | Sitta carolinensis | 100 | 100 | yes | inaturalist |
| American Robin | Turdus migratorius | 100 | 100 | yes | inaturalist |
| Gray Catbird | Dumetella carolinensis | 100 | 100 | yes | inaturalist |
| Song Sparrow | Melospiza melodia | 100 | 100 | yes | inaturalist |
| Northern Cardinal | Cardinalis cardinalis | 100 | 100 | yes | inaturalist |
| Red-winged Blackbird | Agelaius phoeniceus | 100 | 100 | yes | inaturalist |
| Common Grackle | Quiscalus quiscula | 100 | 100 | yes | inaturalist |
| Baltimore Oriole | Icterus galbula | 100 | 100 | yes | inaturalist |
| American Goldfinch | Spinus tristis | 100 | 100 | yes | inaturalist |
| House Sparrow | Passer domesticus | 100 | 100 | yes | inaturalist |
| Total |  | 2000 | 2000 |  |  |

Kept split sizes: train 2400, val 600, calib 400, test 600.

Rerun from this folder with `python download_dataset.py`.
