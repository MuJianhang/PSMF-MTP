# PSMF-MTP
<<<<<<< HEAD
=======

This repository releases the dataset, eight extracted physicochemical/structural feature groups, the pretrained sequence feature representation model, and several non-core utility scripts for multi-functional therapeutic peptide prediction.

The core model training code and final model weights are not included in this release.

## Repository structure

```text
PSMF-MTP/
  dataset/
    train_new.fasta
    test_new.fasta

  features/
    AAC/
      train_aac.csv
      test_AAC.csv
    DDE/
      train_dde.csv
      test_dde.csv
    CKSAAGP/
      train_cksaagp.csv
      test_cksaagp.csv
    PAAC/
      train_paac.csv
      test_paac.csv
    APAAC/
      train_apaac.csv
      test_apaac.csv
    AAindex1/
      aaindex1.my.csv
      train_aaindex1.npy
      test_aaindex1.npy
    CTD/
      train_ctd.npy
      test_ctd.npy
    MorganFP/
      train_morganfp_400.npy
      test_morganfp_400.npy
      morganfp_info.txt
      morganfp_pca400.pkl
      morganfp_scaler.pkl

  pre_trained_model/
    config.json
    model.safetensors
    training_args.bin
    vocab.json

  scripts/
    read_fasta_labels.py
    load_features_demo.py

  requirements.txt
  README.md
```

## Dataset

The dataset is provided in FASTA format:

```text
dataset/train_new.fasta
dataset/test_new.fasta
```

Each FASTA header contains a 21-dimensional binary label vector.

Example:

```text
>000000000000000010000
ARRRRCSDRFRNCPADEALCGRRRR
```

The 21 label order is:

```text
AAP, ABP, ACP, ACVP, ADP, AEP, AFP, AHIVP, AHP, AIP, AMRSAP, APP, ATP, AVP, BBP, BIP, CPP, DPPIP, QSP, SBP, THP
```

## Feature files

Eight feature groups are released:

| Feature group | Released file type | Raw dimension | Final dimension |
|---|---:|---:|---:|
| AAC | CSV | 20 | 400 |
| DDE | CSV | 400 | 400 |
| CKSAAGP | CSV | 100 | 400 |
| PAAC | CSV | 22 | 396 |
| APAAC | CSV | 24 | 384 |
| AAindex1 | NPY | 566 | 566 |
| CTD | NPY | 441 | 441 |
| MorganFP | NPY | 400 | 400 |

The first five CSV feature groups are released in their original extracted dimensions. The utility script `load_features_demo.py` expands them according to the paper and concatenates all eight feature groups into a 3387-dimensional representation.

Final feature order:

```text
AAC:       0-400
DDE:       400-800
CKSAAGP:   800-1200
PAAC:      1200-1596
APAAC:     1596-1980
AAindex1:  1980-2546
CTD:       2546-2987
MorganFP:  2987-3387
```

## Pretrained model

The released pretrained sequence feature representation model is stored in:

```text
pre_trained_model/
```

The model was pretrained on UniRef90 short peptide sequences with lengths of 1-50 aa using the MLM task. The released files correspond to the 30k version used in the paper.

Released files:

```text
config.json
model.safetensors
training_args.bin
vocab.json
```

## Utility scripts

### 1. Parse FASTA labels

```bash
python scripts/read_fasta_labels.py
```

This script reads:

```text
dataset/train_new.fasta
dataset/test_new.fasta
```

and exports:

```text
parsed_labels/train_sequences_labels.csv
parsed_labels/test_sequences_labels.csv
```

### 2. Load and concatenate eight feature groups

```bash
python scripts/load_features_demo.py
```

This script reads the eight feature groups under:

```text
features/
```

and exports:

```text
features/merged/train_features_3387.npy
features/merged/test_features_3387.npy
```

Expected output shapes:

```text
Training features: (7872, 3387)
Test features:     (1969, 3387)
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

The two provided utility scripts only require `numpy` and `pandas`. Other dependencies are included for loading the released pretrained model and preprocessing files if needed.

## Notes

This release includes dataset files, extracted features, the pretrained sequence feature representation model, and non-core utility scripts. The final model training code, core model architecture, hyperparameter tuning scripts, and final trained prediction weights are not included in this release.
>>>>>>> 0d42e6f (ffirst conmit)
