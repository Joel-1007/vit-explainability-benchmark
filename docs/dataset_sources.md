# Dataset sources and what to obtain

Last checked: 29 September 2026. These are publisher-maintained pages or links supplied by the dataset authors. I checked that the public landing pages resolve and describe the files below; I did **not** download any dataset or verify an authenticated download end to end. ImageNet and Stanford access can require an account, acceptance of terms, or approval. Large archive links may redirect to short-lived signed URLs, so use the stable landing page if a direct archive link stops working.

## Which dataset plan does this document cover?

The [project implementation guide](implementation_guide.md) calls for four domains: ImageNet-1K, CUB-200-2011, PASCAL VOC 2012 segmentation, and a CheXpert subset for five findings. To evaluate *localization* (whether an explanation highlights the right part of an image), images and class labels alone are insufficient. The complete guide-aligned acquisition set is:

1. **ImageNet-1K** images/labels **plus ImageNet-S-50** splits and pixel masks for localization.
2. **CUB-200-2011** images/class/part/bounding-box annotations **plus the separate segmentation archive**.
3. **VOC2012 train/validation archive**, using only its segmentation split and masks; optionally add SBD if reproducing the larger `trainaug` split.
4. **Original CheXpert v1.0** images/labels, filtered to the five findings, **plus CheXlocalize** radiologist localization annotations.

The present repository instead configures **NIH ChestX-ray14** as its fourth dataset in [`configs/nih_chestxray.yaml`](../configs/nih_chestxray.yaml) and [`scripts/verify_datasets.py`](../scripts/verify_datasets.py). NIH is documented separately below for running the *current* project. Downloading CheXpert/CheXlocalize does not make the current NIH pipeline use them; code/configuration changes are still needed. Likewise, acquiring data alone does not fix the current dataset loaders or validate the localization results.

## 1. ImageNet-1K + ImageNet-S-50

**What it is and why it is needed.** ImageNet-1K (ILSVRC 2012) supplies the broad 1,000-class natural-image classification data used for the ImageNet experiment and as the image source for ImageNet-S-50. ImageNet-S-50 selects 50 of those classes and supplies segmentation annotations, allowing an attribution map to be compared against the actual object pixels. The base ImageNet archive by itself does **not** provide the ImageNet-S masks.

| Link | What it provides | Project use |
| --- | --- | --- |
| [ImageNet official download page](https://www.image-net.org/download.php) | Publisher entry point and access terms; directs ILSVRC data users to Kaggle. | Establishes the authorized ImageNet-1K source. |
| [ILSVRC/Kaggle data page](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) | Full challenge package: training/validation images and ground truth, test images, XML localization annotations, label mapping and solution files. Requires Kaggle access/competition rules. | Obtain the **complete** ImageNet-1K train/validation images and labels; its boxes are an additional coarse localization reference. Do not substitute a tiny sample or validation-only package. |
| [ImageNet-S official repository and preparation instructions](https://github.com/LUSSeg/ImageNet-S) | Defines the 50-class split, ImageNet-to-ImageNet-S image mapping, annotation layout, and mask encoding. | Assemble ImageNet-S-50 from ImageNet-1K, pairing each image with its correct mask. |
| [ImageNet-S-50 annotation archive](https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip) ([author's download script](https://github.com/LUSSeg/ImageNet-S/blob/main/datapreparation/datapreparation_anno.sh)) | Separate segmentation annotation archive, including validation segmentation masks and the small annotated training subset. It is **not** a replacement for the ImageNet images. | Pixel-level ground truth for localization metrics. |

For reproducible local localization scores, use the ImageNet-S-50 **validation images and `validation-segmentation` masks**. Its test images do not come with public test masks; test evaluation uses the authors' online server. The authors list 64,431 train, 752 validation and 1,682 test images for S-50. Decode mask IDs as `R + 256*G`; `1000` is ignore and `0` is “other.” Keep the ImageNet synset-to-class mapping rather than assuming folder order matches mask IDs.

## 2. CUB-200-2011 + segmentation masks

**What it is and why it is needed.** CUB contains 11,788 images of 200 bird species. Fine-grained explanations should highlight diagnostic bird regions, not just background. The main archive gives class labels, official train/test split, bird bounding boxes and **15 part-location points**. Points support a part-hit/proximity test; they are not area masks. The separate segmentation archive gives bird silhouettes for pixel-overlap metrics such as IoU.

| Link | What it provides | Project use |
| --- | --- | --- |
| [Caltech CUB dataset overview](https://www.vision.caltech.edu/datasets/cub_200_2011/) | Publisher description and links to both required records. | Check usage restrictions and provenance. |
| [CUB images and annotations record](https://data.caltech.edu/records/65de6-vp158) ([`CUB_200_2011.tgz` download](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1)) | All images, species labels, split files, bounding boxes, attributes and part locations. Published MD5: `97eceeb196236b17998738112f37df78`. | Classification/fine-tuning and box/part localization. |
| [CUB segmentations record](https://data.caltech.edu/records/w9d68-gec53) ([`segmentations.tgz` download](https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1)) | Separate foreground masks for the bird images. Published MD5: `4d47ba1228eae64f2fa547c47bc65255`. | Pixel-level localization comparisons; required by this repository's dataset verifier. |

After extraction, align masks by image-relative path under `CUB_200_2011/segmentations/`. Preserve the provided image IDs and split files. This repository also creates a validation split from the CUB training portion; it should not use the official test images for tuning.

## 3. PASCAL VOC 2012 segmentation subset

**What it is and why it is needed.** VOC2012 has 20 object classes and scenes that may contain multiple classes. Its semantic masks give an object-level reference for comparing explanations against the target class. “Segmentation subset” means **select the images listed in `ImageSets/Segmentation/train.txt` and `val.txt` from the full VOC2012 train/validation archive**; it is not a separate official download.

| Link | What it provides | Project use |
| --- | --- | --- |
| [VOC2012 official page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/) | Publisher description, terms, development-kit links and split definitions. | Authoritative source and protocol. |
| [VOC2012 training/validation archive](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | `VOCdevkit/VOC2012` with JPEG images, `SegmentationClass` masks, image-set lists, object annotations and other tasks. | **Required** images and pixel masks for the official segmentation train/validation split (1,464 train and 1,449 val images). |
| [Semantic Boundaries Dataset (SBD) archive](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) ([Berkeley directory](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/)) | Extra VOC-compatible segmentation annotations from a different dataset. | **Optional** only if implementing an augmented `trainaug` protocol such as the 10,582-image training split; not needed for the official VOC2012 segmentation split. |

Use the class-indexed `SegmentationClass` masks for class-specific overlap; `SegmentationObject` answers a different, instance-level question. VOC2012 public test ground-truth masks are not released, so local localization evaluation should use validation. The repository's current benchmark text mentions larger counts without a fully specified SBD integration; do not label that as the official VOC segmentation split.

## 4. Medical subset: CheXpert + CheXlocalize

**What it is and why it is needed.** The intended task is a **five-label, multi-label** chest-X-ray experiment: Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion. Original CheXpert v1.0 supplies radiographs and observation labels (including uncertain labels) for model training/evaluation. It does **not** supply finding-location masks. CheXlocalize supplies radiologist-drawn pixel segmentations and representative points for a smaller, matching CheXpert evaluation subset, so localization can be measured against human annotations. Filter both resources to the same five findings; do not treat the CheXlocalize model-generated Grad-CAM maps as ground truth.

| Link | What it provides | Project use |
| --- | --- | --- |
| [Original CheXpert v1.0 dataset page](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) ([Stanford access portal](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)) | Full original chest-radiograph dataset (224,316 images from 65,240 patients) and image-level observation labels. Access is through Stanford's portal and its terms. | Obtain images and labels, then select the **five columns** above; retain patient identifiers/splits and explicitly handle uncertain labels. |
| [CheXpert task/label description](https://stanfordmlgroup.github.io/competitions/chexpert/) | Original challenge description, label and evaluation context. | Choose a defensible five-finding label/uncertainty policy and cite the correct version. |
| [CheXlocalize official dataset page](https://aimi.stanford.edu/datasets/chexlocalize) ([Stanford download portal](https://stanford.redivis.com/datasets/efx9-5nspnbb4b)) | Localized CheXpert radiographs plus radiologist annotations for ten findings, including the required five. Access may require registration/terms. | Ground-truth localization masks/points for a held-out medical evaluation subset. |
| [CheXlocalize authors' code/data-layout guide](https://github.com/rajpurkarlab/cheXlocalize) | Explains the `gt_segmentations_val.json` / `gt_segmentations_test.json` masks, raw annotations and image layout. | Pair the five selected finding labels with the correct image and radiologist mask; understand mask encoding. |

The five-finding task is a **selection from CheXpert**, not a separately downloadable “CheXpert 5” dataset. For full training, request the original CheXpert dataset, not merely a small sample or **CheXpert Plus** (a distinct release). Reserve CheXlocalize for localization evaluation and maintain patient-level separation from training. “Radiologist attention heatmaps” in the implementation guide should be read as **radiologist localization annotations** here; attention/Grad-CAM heatmaps are model outputs, not the human reference standard.

## If running the repository unchanged: NIH ChestX-ray14 instead of CheXpert

The present verifier and configuration expect NIH ChestX-ray14, not CheXpert. The [NIH Clinical Center distribution](https://nihcc.app.box.com/v/ChestXray-NIHCC) is the source named by the [dataset authors' paper](https://arxiv.org/abs/1705.02315). Obtain **all** image archives (112,120 PNGs), `Data_Entry_2017_v2.csv` (or reconcile the distributed metadata filename with the verifier), `train_val_list.txt`, `test_list.txt`, and `BBox_List_2017.csv`; the last file provides boxes for a limited pathology subset. The NIH Box page could not be checked anonymously in this environment, so its present download availability is **unconfirmed** here; do not assume an individual file URL will work without visiting the Box page.

NIH is **not interchangeable** with the guide's five CheXpert findings. Its labels, cohort, and localization subset differ; importantly, its box subset does not provide the same five-condition pixel masks as CheXlocalize. Choose one medical protocol and update the code, metrics, documentation and reported results consistently before interpreting cross-dataset comparisons.

## Before acquiring or using the data

- Accept each publisher's terms. Do not commit images, credentials, signed download links or patient data to this repository.
- Keep the complete publisher archives/records and record their versions, checksums where provided, and exact train/validation/test IDs. Do not infer an evaluation split from file order.
- Classification labels tell **what** is present; boxes, part points and segmentation masks tell **where**. Use only examples with the relevant spatial annotation when computing localization metrics.
- The links above were verified as public source/record pages or author-published archive URLs, **not** by transferring the archives. Account-gated payloads and large-file redirects must be confirmed after authorized access.
