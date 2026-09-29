# Using the dataset download script

This guide explains how to run [`scripts/download_datasets.sh`](../scripts/download_datasets.sh) on a Linux server or in Git Bash on Windows. For the publisher links, access conditions, and why each dataset is needed, see [Dataset sources](dataset_sources.md).

The script **downloads source archives/files only** into a directory you choose. It does not extract archives, prepare the benchmark's loader layout, filter CheXpert to five findings, or make the repository's current NIH configuration use CheXpert. A successful download is not a completed dataset validation.

## Before running it

1. Clone this repository on the machine that will store the data, and run the commands from its root directory. Choose an existing storage volume with ample free space. Pass an **absolute** `--data-root` path to a specific directory, not `/`, a drive root, or your home directory. The script writes to `DATA_ROOT/_archives/`.
2. Confirm the dataset owners' access requirements and terms before transfer. A direct URL responding without a login does not itself grant permission to use the dataset. The ImageNet Kaggle route requires the server operator's own authenticated Kaggle CLI and accepted competition rules; `--imagenet-direct` avoids installing Kaggle **on that server**, not the publisher's access conditions.
3. Install Bash, `curl`, `tar`, `unzip`, and `md5sum`. Probes require **curl 8.4+** to cap received bytes safely. `--chexlocalize-api` additionally requires Python 3 and curl 8.4+. The Kaggle ImageNet route requires the `kaggle` CLI; the direct route does not.
4. If using `--chexlocalize-api`, provide an authorized Redivis token as `REDIVIS_API_TOKEN` in the process environment, or place `REDIVIS-DATA-API-KEY=...` in an **untracked, protected** `.env` at the repository root. The helper reads that entry as data; it does not execute `.env`. On a shared server, use an approved secret-injection method and coordinate with the owner before placing a personal token there. Do not put tokens or temporary signed URLs in commands, logs, or Git.

## Typical commands

Check the available options without transferring data:

```bash
bash scripts/download_datasets.sh --help
```

For the guide-aligned sources, first make small network probes. If they pass and you have permission and storage, run the same selection **without** `--probe` to start full transfers:

```bash
bash scripts/download_datasets.sh --data-root /mnt/research-data --all --imagenet-direct --chexpert-direct --chexlocalize-api --probe
bash scripts/download_datasets.sh --data-root /mnt/research-data --all --imagenet-direct --chexpert-direct --chexlocalize-api
```

These commands select ImageNet-1K images and ImageNet-S-50 annotations, CUB images and masks, VOC2012 train/validation, the CheXpert v1.0 ZIP, and the official Redivis CheXlocalize file index. The full transfer is **hundreds of gigabytes** before extraction, with the CheXpert ZIP alone approximately 471 GB according to the endpoint headers. Do not run the second command merely to test links.

To work with one family at a time, for example:

```bash
bash scripts/download_datasets.sh --data-root /mnt/research-data --cub --probe
bash scripts/download_datasets.sh --data-root /mnt/research-data --cub
bash scripts/download_datasets.sh --data-root /mnt/research-data --chexlocalize-api --probe
bash scripts/download_datasets.sh --data-root /mnt/research-data --chexlocalize-api
```

`--chexlocalize-api` uses the Redivis token. Its files contain localization annotations and model-generated maps, **not** CheXpert X-rays or label CSVs. For validation localization, pair its human annotations with CheXpert validation images/labels. The separately distributed test X-rays/labels are needed only if you evaluate the CheXlocalize test split.

On **Windows without WSL**, use Git Bash (from Git for Windows) and Git Bash paths, not PowerShell paths:

```bash
cd /c/Users/Jayan/Coding/Projects/Research/vit-explainability-benchmark
bash scripts/download_datasets.sh --data-root /d/research-data --cub --probe
```

Replace `/d/research-data` with a real destination on your machine. A Linux server uses its own absolute path, such as `/mnt/research-data`; `/mnt/...` is not a native Git Bash drive path. The script is Bash-based, so it is not directly runnable in PowerShell without a Bash environment.

## Option reference

| Option | Effect |
| --- | --- |
| `--data-root PATH` | Required absolute destination; saves under `PATH/_archives/`. |
| `--all` | Selects ImageNet, CUB, VOC, and **medical access notes**. It does not by itself fetch CheXpert or CheXlocalize. |
| `--imagenet` | Uses the Kaggle ImageNet competition route plus the separate ImageNet-S-50 annotation archive. |
| `--imagenet-direct` | Selects ImageNet using publisher-hosted image, devkit, and bounding-box archives instead of Kaggle, plus ImageNet-S-50. |
| `--cub` | Downloads CUB-200-2011 and its separate segmentation masks. |
| `--voc` | Downloads the official VOC2012 train/validation archive. The segmentation subset is selected later from its split files. |
| `--sbd` | Adds optional SBD annotations for an augmented VOC protocol; requires `--voc` or `--all`. |
| `--medical` | Prints CheXpert/CheXlocalize access notes only. |
| `--chexpert-direct` | Selects the full publisher-hosted CheXpert v1.0 ZIP. Confirm authorized use first. |
| `--chexlocalize-api` | Selects all files in the pinned, authorized CheXlocalize Redivis v1.0 index; requires a token. |
| `--nih` | Prints NIH ChestX-ray14 access notes for the repository's *current* medical pipeline; does not download NIH. |
| `--probe` | Uses the network to test small byte ranges or small metadata for the selected sources, without saving full archives. |
| `--dry-run` | Legacy alias for `--probe`; **not** an offline or zero-download dry run. |

## What the script saves and checks

| Selection | Output under `DATA_ROOT/_archives/` |
| --- | --- |
| ImageNet | `imagenet/` for ImageNet archives; `imagenet_s50/` for the segmentation archive. |
| CUB | `cub200/` for the main dataset and segmentation archives. |
| VOC, optional SBD | `voc/`. |
| CheXpert | `chexpert/CheXpert-v1.0.zip`. |
| CheXlocalize | `chexlocalize/CheXlocalize/` with the Redivis-indexed paths. |

In probe mode, direct archive sources request up to the first 4 KiB and require a partial-content response; no archive is saved. The authenticated Redivis probe checks table metadata and one small file range. The default Kaggle ImageNet probe is different: it **saves** a small class-mapping file under `_archives/_probe/imagenet/` to test account access. A probe can pass even if a later full transfer fails or files from two sources do not match.

In full mode, direct archive transfers use `.part` files so interrupted transfers can resume. Completed archives are tested with `tar` or `unzip`; publisher MD5 values are checked where the script has them. The **Kaggle route** currently checks that the CLI saved a non-empty file, but does not individually validate every Kaggle archive. The Redivis helper verifies each completed file's size and indexed MD5, and can skip verified files on rerun. A corrupted *completed* file causes an error for investigation rather than being silently overwritten. These are transfer-integrity checks, **not** checks of extracted file counts, labels, or image-to-mask matches. A large ZIP integrity test may itself take substantial time.

If a probe says `FAIL`, fix the reported tool, authorization, or connection issue before full download. `UNVERIFIED` means the host did not demonstrate a safe ranged response; do not interpret it as a pass. If a full transfer is interrupted, rerun the same command with the same data root to resume where supported. If it reports an existing corrupt archive or a checksum mismatch, inspect the exact file and source before any manual removal; do not assume the benchmark data is valid.

## After downloading

The script does **not** extract or reorganize archives. Keep the original archives, record versions and checksums, then prepare the loader-specific directories and split files separately. The current [`verify_datasets.py`](../scripts/verify_datasets.py) covers some existing datasets but not CheXpert/CheXlocalize. [Post-extraction, cross-dataset validation remains a TODO](dataset_sources.md#todo-validate-prepared-datasets-after-download-and-extraction): check image counts, label CSVs, split/patient separation, and that every selected CheXlocalize annotation matches the intended CheXpert image and finding before reporting localization results.
