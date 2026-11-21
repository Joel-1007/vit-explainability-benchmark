# Server dataset setup

The benchmark uses **CUB-200-2011, PASCAL VOC 2012, ImageNet-S-50 and NIH ChestX-ray14**. ImageNet-1K supplies the source images for S-50. Prepare both ImageNet training and validation images to fine-tune on S-50 and then evaluate localization.

## Source links and verification

Checked on 21 November 2026. The checks covered public pages and publisher archive listings. Archive transfers, signed-in access and checksums still need verification after acquisition.

| Dataset | Where to acquire it | What the link check established |
| --- | --- | --- |
| ImageNet-1K (ILSVRC 2012) | [Kaggle competition data](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data), linked by the [ImageNet organizers](https://www.image-net.org/challenges/LSVRC/) | Kaggle returned the competition page. Sign in and accept the competition rules to access the files. The PDF's bare ImageNet homepage timed out during the live check. |
| ImageNet-S-50 | [Authors' preparation instructions](https://github.com/LUSSeg/ImageNet-S#imagenet-s-dataset-preparation) and [S-50 annotation ZIP](https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip) | The authors' repo loaded and the ZIP was listed on the official release. |
| CUB-200-2011 | [Caltech dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/), [image archive record](https://data.caltech.edu/records/65de6-vp158), [mask archive record](https://data.caltech.edu/records/w9d68-gec53) | Both CaltechDATA records loaded and list the archives and their MD5 checksums. |
| PASCAL VOC 2012 | [Oxford development kit](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit), then **training/validation data** | Oxford's page loaded and lists the train/validation TAR. |
| NIH ChestX-ray14 | [NIH Clinical Center Box folder](https://nihcc.app.box.com/v/ChestXray-NIHCC) or the [Google Cloud PNG source](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#cloud-storage) | Google documents the PNG bucket and links the NIH Box folder. Box returned its JavaScript shell. Check its file list in a regular browser. Google Cloud documents Requester Pays billing. |

The PDF's secondary references also resolved. [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/voc) and [GluonCV](https://cv.gluon.ai/build/examples_datasets/pascal_voc.html) are VOC references. [ImageNetSegModel](https://github.com/LUSSeg/ImageNetSegModel) is a model repository and [Intel's ImageNet notes](https://github.com/intel/ai-reference-models/blob/main/datasets/imagenet/README.md) describe dataset setup. The [NIH Kaggle listing](https://www.kaggle.com/datasets/nih-chest-xrays/data) resolved, but its file inventory could not be checked in the public link check.

## Storage on the server

Use a shared filesystem visible from the compute nodes. The examples use `/mnt/research-data/vit-benchmark` as `DATA_ROOT`. Store source downloads under `$DATA_ROOT/_archives/`, then extract or link the prepared data into the layout below. Keep the publisher filenames, directory names, and case. Budget for archives, extracted files, model weights and benchmark outputs.

After accepting the dataset terms and configuring Kaggle access, the repository's [download helper](dataset_download_guide.md) can fetch the ImageNet, CUB and VOC archives. For this NIH-based run, select `--imagenet --cub --voc` explicitly and acquire NIH from Box or Google separately. The helper writes archives only. Continue with the extraction and layout steps below.

```text
$DATA_ROOT/
  _archives/
    cub200/
    voc/
    imagenet/
    imagenet_s50/
    nih_chestxray/
  imagenet_1k/                        # staging for ImageNet-S preparation
    train/<wnid>/*.JPEG
    val/<wnid>/*.JPEG
  imagenet_s50/
    train/<wnid>/*.JPEG
    validation/<wnid>/*.JPEG
    validation-segmentation/<wnid>/*.png
  cub200/
    images/<class>/*.jpg
    segmentations/<class>/*.png
    images.txt
    image_class_labels.txt
    train_test_split.txt
    bounding_boxes.txt
  voc2012/
    JPEGImages/
    Annotations/
    SegmentationClass/
    ImageSets/Main/train.txt
    ImageSets/Main/val.txt
    ImageSets/Segmentation/train.txt
    ImageSets/Segmentation/val.txt
  nih_chestxray/
    images/*.png                      # flat directory, original NIH filenames
    Data_Entry_2017_v2.csv              # accepted alternatives below
    BBox_List_2017.csv
    train_val_list.txt
    test_list.txt
```

### ImageNet-1K and ImageNet-S-50

1. Acquire the ILSVRC 2012 **training and validation JPEGs** and the validation-to-synset mapping. The Kaggle distribution usually extracts under `ILSVRC/Data/CLS-LOC/{train,val}`. If validation JPEGs are in a flat folder, use the official validation labels with the devkit synset metadata, or the competition's validation mapping, to create the `val/<wnid>/` layout. Preserve the basename for each JPEG. If using the original training TAR, unpack the per-synset TARs into separate `train/<wnid>/` folders.
2. Use the [authors' preparation tools](https://github.com/LUSSeg/ImageNet-S/tree/main/datapreparation) from their `datapreparation/` directory. Use the **50-class** option and an absolute path to the prepared ImageNet-1K root. The training tool links to the source images by default. Keep the source files on a stable mount, or request copies if the prepared tree needs to move independently. The validation tool prepares the S-50 validation and test image folders from ImageNet-1K validation using the supplied mappings. Some validation images change synset folders during this step, so preserve the [authors' source-to-destination mapping](https://github.com/LUSSeg/ImageNet-S/blob/main/data/mapping/ImageNetS_im50_mapping_validation.txt).
3. Extract the S-50 annotation ZIP alongside the generated `ImageNetS50` directory so `validation-segmentation/` sits alongside `validation/`. Place or link the resulting `ImageNetS50` root at `$DATA_ROOT/imagenet_s50`. Check **64,431 training images**, **752 validation images**, 50 matching synset directories, and one PNG per validation image, matched by synset and stem. The authors hold the test masks on the evaluation server. This benchmark evaluates the published validation masks locally.
4. The training code makes a deterministic holdout from the S-50 training set using split seed `0`. Preserve the official S-50 validation set for benchmark evaluation.

### CUB-200-2011

1. Obtain `CUB_200_2011.tgz` and `segmentations.tgz` from the two CaltechDATA records. Published MD5 values are `97eceeb196236b17998738112f37df78` and `4d47ba1228eae64f2fa547c47bc65255`, respectively.
2. Extract both archives. Place or link the directory containing `images.txt` at `$DATA_ROOT/cub200`. Put the separate `segmentations/` directory inside the same root, preserving the class subfolders and file stems.
3. Expect **11,788 images and masks**. Keep the publisher's split, labels, boxes and part annotations. Fine-tuning takes its holdout from the official training set. Evaluation uses the official test set.

### PASCAL VOC 2012

1. Download `VOCtrainval_11-May-2012.tar` from Oxford's **Development Kit** section and extract it. Place or link its `VOCdevkit/VOC2012` directory at `$DATA_ROOT/voc2012`.
2. Retain the full `JPEGImages/` and `Annotations/` directories and both `ImageSets/Main/` and `ImageSets/Segmentation/`. Keep the indexed PNGs under `SegmentationClass/`. Any library-based download must preserve those files and the original split lists.
3. Fine-tuning reads the Main training split and XML labels, and uses the Main validation split with segmentation validation IDs excluded for threshold fitting. Localization evaluates the **1,449 official segmentation validation images**, one unit per labeled class per image. The official segmentation training list contains **1,464 images**. SBD is optional for a separate augmented training protocol.

### NIH ChestX-ray14

1. In the NIH Box folder, obtain all image parts and the publisher metadata. With the Google Cloud alternative, use the original PNGs in `gs://gcs-public-data--healthcare-nih-chest-xray/png/` and obtain the split and box metadata separately from NIH. Keep the names of the original PNGs.
2. Extract and consolidate all **112,120 PNGs** directly under `$DATA_ROOT/nih_chestxray/images/`. Keep the CSV and split lists in the parent directory. The label file can be `Data_Entry_2017_v2.csv`, `Data_Entry_2017.csv`, or `Data_Entry_2017_v2020.csv` if it has the required `Image Index` and `Finding Labels` columns. Keep `train_val_list.txt`, `test_list.txt`, and `BBox_List_2017.csv`.
3. The box CSV contains **984 boxes across 880 images** before the patient exclusions. Fine-tuning holds out patients within the official training pool, and the verifier checks the actual training, threshold-validation and localization splits for patient separation. Localization uses the eligible expert boxes. Full test images and the split lists remain part of the dataset integrity check.

### CheXlocalize and the Redivis token

The four datasets above require no Redivis token. The older CheXpert protocol would require [CheXpert radiographs and labels](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) plus matching radiologist annotations from the [CheXlocalize Redivis files](https://stanford.redivis.com/datasets/efx9-5nspnbb4b/files). The Redivis download API needs dataset access and a token with the `data.data` scope. If preparing that separate protocol, provide the token to the existing helper through `REDIVIS_API_TOKEN`. See the [Redivis access requirements](https://docs.redivis.com/api/rest-api/files/get#authorization) and the [CheXlocalize authors' layout](https://github.com/rajpurkarlab/cheXlocalize).

## Preflight once execution is authorized

Run from the repository root, after acquisition and extraction. The following commands are for the server setup, and were not executed during the review.

```bash
export DATA_ROOT=/mnt/research-data/vit-benchmark
uv run python scripts/verify_datasets.py \
  --cub "$DATA_ROOT/cub200" \
  --voc "$DATA_ROOT/voc2012" \
  --imagenet_s "$DATA_ROOT/imagenet_s50" \
  --nih "$DATA_ROOT/nih_chestxray"
uv run python scripts/audit_datasets.py --data-root "$DATA_ROOT" --output_dir results/data-audit
```

1. Confirm the server driver supports the CUDA runtime selected by `uv.lock`. Use the [NVIDIA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).
2. Fine-tune or stage the required weights. The main matrix uses **72 final checkpoints** for six models, four datasets and seeds `0`, `1`, `2`. The VOC and NIH weights must include the validation-fitted decision thresholds. See the [fine-tuning and seed options](../README.md#quickstart).
3. Run a small CUB pilot with a real checkpoint before launching the full run. Set `WEIGHTS_DIR` and `PILOT_FINETUNE_SEED` to select the checkpoint. To include BEiT and DINOv2 in the smoke checks, set `PILOT_SMOKE_MODELS="vit_b16 beit_b16 dinov2_vitb14"` and stage CUB checkpoints for all three at that seed.
4. Use a fresh result directory for the first full run under this version of the code. Keep one writer per result directory and keep the dataset, model files and software fixed while resuming. Resume compares the evaluated sample list, checkpoint path, size and modification time, device and recorded settings. Use a fresh directory when replacing dataset files in place, altering code or switching between pilot and full settings.
5. Launch the full benchmark with `FINETUNE_SEEDS="0 1 2"` once the preflight passes. Check `coverage.csv` for the completed and explicitly unsupported combinations, and `seed_summary.csv` for the number of seeds represented.
