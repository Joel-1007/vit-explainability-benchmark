#!/usr/bin/env bash
# Download the source archives for the guide-aligned datasets to a chosen server.
# This does not extract data or bypass dataset access/usage requirements.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/download_datasets.sh --data-root /absolute/path [--all | --imagenet] [--imagenet-direct] [--cub] [--voc] [--sbd] [--medical] [--chexpert-direct] [--chexlocalize-api] [--nih] [--probe | --dry-run]

  --all       ImageNet-1K + ImageNet-S-50 annotations, CUB, VOC, medical access notes
  --imagenet  Kaggle ImageNet-1K package + ImageNet-S-50 annotation archive
  --imagenet-direct  Use the publisher-hosted ImageNet archives instead of Kaggle
  --cub       CUB images/annotations + separate segmentation masks
  --voc       Official VOC2012 train/validation archive
  --sbd       Optional SBD augmentation (requires --voc or --all)
  --medical   Show CheXpert and CheXlocalize access steps
  --chexpert-direct  Fetch the full Stanford-hosted CheXpert v1.0 ZIP (~471 GB):
              train/validation images and labels (confirm contents after extraction)
  --chexlocalize-api  Use an authorized Redivis API token to fetch every file in
              the official CheXlocalize v1.0 file index (~3.9 GB; annotations
              and model maps, but no X-rays or label CSVs)
  --nih       Show current-code NIH ChestX-ray14 access steps (alternative to CheXpert)
  --probe     Test public archive links with a bounded 4 KiB range request;
              download Kaggle's small class-mapping file instead of ImageNet images
  --dry-run   Alias for --probe (kept for existing commands; it now uses the network)

Downloads go under DATA_ROOT/_archives/. Extraction and dataset preparation are
separate steps; these archives are not yet in the layout expected by the loaders.
EOF
}

data_root=""
all=false
imagenet=false
imagenet_direct=false
cub=false
voc=false
sbd=false
medical=false
chexpert_direct=false
chexlocalize_api=false
nih=false
probe=false
probe_passes=0
probe_failures=0
probe_inconclusive=0

while (($#)); do
  case "$1" in
    --data-root)
      if (($# < 2)); then echo 'Missing value for --data-root' >&2; exit 2; fi
      data_root=$2
      shift 2
      ;;
    --all) all=true; shift ;;
    --imagenet) imagenet=true; shift ;;
    --imagenet-direct) imagenet=true; imagenet_direct=true; shift ;;
    --cub) cub=true; shift ;;
    --voc) voc=true; shift ;;
    --sbd) sbd=true; shift ;;
    --medical) medical=true; shift ;;
    --chexpert-direct) medical=true; chexpert_direct=true; shift ;;
    --chexlocalize-api) medical=true; chexlocalize_api=true; shift ;;
    --nih) nih=true; shift ;;
    --probe|--dry-run) probe=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$data_root" || "$data_root" != /* || "$data_root" == / ]]; then
  echo 'Pass an explicit absolute --data-root other than /.' >&2
  exit 2
fi
if [[ "$data_root" == "$HOME" ]]; then
  echo 'Choose a specific data directory, not the home directory itself.' >&2
  exit 2
fi
if [[ "$all" == true ]]; then
  imagenet=true
  cub=true
  voc=true
  medical=true
fi
if [[ "$sbd" == true && "$voc" != true ]]; then
  echo '--sbd requires --voc or --all.' >&2
  exit 2
fi
if [[ "$imagenet" != true && "$cub" != true && "$voc" != true && "$medical" != true && "$nih" != true ]]; then
  usage >&2
  exit 2
fi

archive_root="${data_root%/}/_archives"
echo "Destination: $archive_root"
echo 'No archives will be extracted by this script.'

probe_curl() {
  local url=$1 result status bytes mime curl_version major minor rc
  if ! command -v curl >/dev/null 2>&1; then
    echo "FAIL: curl is unavailable; cannot test $url" >&2
    probe_failures=$((probe_failures + 1))
    return
  fi
  # Older curl versions may not enforce --max-filesize when size is unknown.
  curl_version=$(curl --version)
  curl_version=${curl_version%%$'\n'*}
  curl_version=${curl_version#curl }
  curl_version=${curl_version%% *}
  major=${curl_version%%.*}
  minor=${curl_version#*.}
  minor=${minor%%.*}
  if [[ ! "$major" =~ ^[0-9]+$ || ! "$minor" =~ ^[0-9]+$ ]] ||
     ((major < 8 || (major == 8 && minor < 4))); then
    echo "FAIL: probing safely requires curl 8.4.0 or newer (found $curl_version)." >&2
    probe_failures=$((probe_failures + 1))
    return
  fi

  echo "Probing first 4 KiB: $url"
  if result=$(curl --silent --show-error --fail --location --max-redirs 10 \
      --range 0-4095 --max-filesize 8192 --connect-timeout 15 --max-time 45 \
      --output /dev/null --write-out '%{http_code} %{size_download} %{content_type}' \
      "$url"); then
    read -r status bytes mime <<< "$result"
    if [[ "$status" == 206 && "$bytes" != 0 && "$bytes" != 0.000 &&
          "$mime" != text/html* ]]; then
      echo "PASS: HTTP 206; received $bytes bytes (no archive saved)."
      probe_passes=$((probe_passes + 1))
    else
      echo "UNVERIFIED: HTTP $status, $bytes bytes, content type ${mime:-unknown}; a ranged archive response was not confirmed." >&2
      probe_inconclusive=$((probe_inconclusive + 1))
    fi
  else
    rc=$?
    if ((rc == 63)); then
      echo 'UNVERIFIED: the host may have ignored the byte range; curl stopped before a large transfer.' >&2
      probe_inconclusive=$((probe_inconclusive + 1))
    else
      echo "FAIL: curl could not fetch the small range (exit $rc)." >&2
      probe_failures=$((probe_failures + 1))
    fi
  fi
}

probe_kaggle() {
  local probe_dir="$archive_root/_probe/imagenet"
  if ! command -v kaggle >/dev/null 2>&1; then
    echo 'FAIL: Kaggle CLI is unavailable; install it, authenticate, and accept the ImageNet competition rules.' >&2
    probe_failures=$((probe_failures + 1))
    return
  fi
  mkdir -p -- "$probe_dir"
  echo 'Probing Kaggle access with LOC_synset_mapping.txt (not the ImageNet images).'
  if kaggle competitions download imagenet-object-localization-challenge \
      -f LOC_synset_mapping.txt -p "$probe_dir" &&
     [[ -s "$probe_dir/LOC_synset_mapping.txt" ]]; then
    echo "PASS: small Kaggle metadata file saved to $probe_dir/LOC_synset_mapping.txt"
    probe_passes=$((probe_passes + 1))
  else
    echo 'FAIL: Kaggle did not provide the metadata file; check authentication and accepted competition rules.' >&2
    probe_failures=$((probe_failures + 1))
  fi
}

download_curl() {
  local url=$1 dest=$2 expected_md5=${3:-}
  if [[ "$probe" == true ]]; then
    probe_curl "$url"
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo 'curl is required for the public archives.' >&2
    exit 1
  fi
  if ! command -v tar >/dev/null 2>&1 || ! command -v unzip >/dev/null 2>&1; then
    echo 'tar and unzip are required to validate downloaded archives.' >&2
    exit 1
  fi
  mkdir -p -- "$(dirname -- "$dest")"
  if [[ -s "$dest" ]]; then
    if [[ -n "$expected_md5" ]]; then
      printf '%s  %s\n' "$expected_md5" "$dest" | md5sum --check --status || {
        echo "Existing archive failed its publisher checksum: $dest" >&2
        exit 1
      }
    fi
    validate_archive "$dest"
    echo "Already present: $dest"
    return
  fi

  echo "Downloading: $url"
  # Keep interrupted transfers in .part; never present a partial file as complete.
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "${dest}.part" "$url"
  if [[ ! -s "${dest}.part" ]]; then
    echo "Download is empty: ${dest}.part" >&2
    exit 1
  fi
  if [[ -n "$expected_md5" ]]; then
    printf '%s  %s\n' "$expected_md5" "${dest}.part" | md5sum --check --status || {
      echo "Publisher checksum failed: ${dest}.part" >&2
      exit 1
    }
  fi
  validate_archive "${dest}.part" "$dest"
  mv -- "${dest}.part" "$dest"
  echo "Saved: $dest"
}

validate_archive() {
  local archive=$1 name=${2:-$1}
  echo "Checking archive: $archive"
  case "$name" in
    *.tar) tar -tf "$archive" >/dev/null ;;
    *.tgz|*.tar.gz) tar -tzf "$archive" >/dev/null ;;
    *.zip) unzip -tq "$archive" >/dev/null ;;
    *) echo "Unknown archive type: $name" >&2; exit 1 ;;
  esac || {
    echo "Archive validation failed: $archive" >&2
    exit 1
  }
}

if [[ "$imagenet" == true ]]; then
  imagenet_dir="$archive_root/imagenet"
  if [[ "$imagenet_direct" == true ]]; then
    echo 'Using publisher-hosted ImageNet files; confirm ImageNet access and terms before transfer.'
    download_curl \
      'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar' \
      "$imagenet_dir/ILSVRC2012_img_train.tar" \
      '1d675b47d978889d74fa0da5fadfb00e'
    download_curl \
      'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar' \
      "$imagenet_dir/ILSVRC2012_img_val.tar" \
      '29b22e2961454d5413ddabcf34fc5622'
    download_curl \
      'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz' \
      "$imagenet_dir/ILSVRC2012_devkit_t12.tar.gz" \
      'fa75699e90414af021442c21a62c3abf'
    download_curl \
      'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz' \
      "$imagenet_dir/ILSVRC2012_bbox_train_v2.tar.gz" \
      '9271167e2176350e65cfe4e546f14b17'
    download_curl \
      'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz' \
      "$imagenet_dir/ILSVRC2012_bbox_val_v3.tgz" \
      'f4cd18b5ea29fe6bbea62ec9c20d80f0'
  else
    if [[ "$probe" == true ]]; then
      probe_kaggle
    else
      if ! command -v kaggle >/dev/null 2>&1; then
        echo 'Install and authenticate the official Kaggle CLI, or use --imagenet-direct after obtaining publisher access.' >&2
        echo 'See https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data' >&2
        exit 1
      fi
      mkdir -p -- "$imagenet_dir"
      kaggle competitions download imagenet-object-localization-challenge -p "$imagenet_dir"
      if [[ -z "$(find "$imagenet_dir" -type f -size +0c -print -quit)" ]]; then
        echo "Kaggle returned without saving any non-empty files in $imagenet_dir" >&2
        exit 1
      fi
    fi
  fi
  download_curl \
    'https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip' \
    "$archive_root/imagenet_s50/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip"
fi

if [[ "$cub" == true ]]; then
  download_curl \
    'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1' \
    "$archive_root/cub200/CUB_200_2011.tgz" \
    '97eceeb196236b17998738112f37df78'
  download_curl \
    'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1' \
    "$archive_root/cub200/segmentations.tgz" \
    '4d47ba1228eae64f2fa547c47bc65255'
fi

if [[ "$voc" == true ]]; then
  download_curl \
    'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar' \
    "$archive_root/voc/VOCtrainval_11-May-2012.tar"
fi

if [[ "$sbd" == true ]]; then
  download_curl \
    'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' \
    "$archive_root/voc/benchmark.tgz"
fi

if [[ "$medical" == true ]]; then
  if [[ "$chexpert_direct" == true ]]; then
    echo 'Using the full Stanford-hosted CheXpert ZIP for train/validation radiographs and labels; confirm Stanford access and terms before transfer.'
    download_curl \
      'https://download.cs.stanford.edu/deep/CheXpert-v1.0.zip' \
      "$archive_root/chexpert/CheXpert-v1.0.zip"
  fi
  if [[ "$chexlocalize_api" == true ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python_bin=python3
    elif command -v python >/dev/null 2>&1; then
      python_bin=python
    else
      echo 'Python 3 is required for the authenticated CheXlocalize REST API transfer.' >&2
      exit 1
    fi
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    redivis_args=(--data-root "$data_root")
    if [[ "$probe" == true ]]; then redivis_args+=(--probe); fi
    "$python_bin" "$script_dir/download_chexlocalize_redivis.py" "${redivis_args[@]}"
  fi
  echo
  echo 'Authorized portal access remains relevant for the guide-aligned medical data:'
  echo '  CheXpert v1.0 terms/portal: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays'
  echo '  CheXlocalize:   https://aimi.stanford.edu/datasets/chexlocalize'
  if [[ "$chexlocalize_api" == true ]]; then
    echo '  Redivis supplies localization annotations/maps, not CheXpert radiographs or label CSVs.'
    echo '  Pair validation annotations with CheXpert validation images/labels; no second validation-image download is needed if they match.'
    echo '  CheXlocalize test annotations require separately obtained, matching test X-rays/labels for test evaluation.'
  else
    echo "  Add --chexlocalize-api for authorized REST API transfer under $archive_root/chexlocalize/."
  fi
  echo '  Do not share login credentials or commit temporary signed URLs to this repository.'
fi

if [[ "$nih" == true ]]; then
  echo
  echo 'The unchanged repository uses NIH ChestX-ray14 instead of CheXpert:'
  echo '  NIH distribution: https://nihcc.app.box.com/v/ChestXray-NIHCC'
  echo '  Obtain all image archives, labels, split lists, and BBox_List_2017.csv.'
  echo "  On the server, store authorized downloads under $archive_root/nih_chestxray/."
  echo '  The NIH Box link could not be anonymously verified; inspect it before relying on it.'
fi

echo
if [[ "$probe" == true ]]; then
  echo "Direct-archive probe results (excluding Redivis): $probe_passes passed; $probe_failures failed; $probe_inconclusive unverified."
  echo 'CheXlocalize REST access was probed only if --chexlocalize-api was selected; NIH was not probed.'
  echo 'A successful probe does not check full-transfer completion, archive integrity, or storage capacity.'
  if ((probe_failures > 0)); then exit 1; fi
  if ((probe_inconclusive > 0)); then exit 3; fi
else
  echo 'Automated download phase finished; portal downloads may still be pending.'
  # TODO: Extend verify_datasets.py to validate extracted datasets and medical image/label/mask alignment.
  echo 'TODO: post-extraction dataset-content and image/label/mask alignment checks are not yet implemented.'
fi
echo 'See docs/dataset_sources.md before extracting or running the benchmark.'
