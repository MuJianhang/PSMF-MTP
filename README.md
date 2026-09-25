# PSMF-MTP

PSMF-MTP is a computational framework for multi-functional therapeutic peptide prediction. This repository provides the benchmark dataset, eight groups of physicochemical and structural features, a short-peptide pretrained sequence encoder, and utility scripts for data parsing, feature preparation, representation extraction, and multi-label evaluation.

## Repository contents

```text
PSMF-MTP/
|-- dataset/
|   |-- train_new.fasta
|   |-- test_new.fasta
|   `-- LABELS.md
|-- features/
|   |-- AAC/
|   |-- DDE/
|   |-- CKSAAGP/
|   |-- PAAC/
|   |-- APAAC/
|   |-- AAindex1/
|   |-- CTD/
|   |-- MorganFP/
|   `-- merged/
|-- pre_trained_model/
|   |-- config.json
|   |-- model.safetensors
|   |-- vocab.json
|   |-- vocab.txt
|   |-- tokenizer_config.json
|   |-- special_tokens_map.json
|   `-- pretraining_metadata.json
|-- scripts/
|   |-- read_fasta_labels.py
|   |-- load_features_demo.py
|   |-- extract_pretrained_features_demo.py
|   |-- evaluate_predictions.py
|   `-- verify_release.py
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.10 or later is recommended. The repository contains large NumPy arrays and model files managed with Git LFS.

```bash
git lfs install
git clone https://github.com/MuJianhang/PSMF-MTP.git
cd PSMF-MTP
git lfs pull
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands below from the repository root.

## Usage

### 1. Verify the downloaded files

```bash
python scripts/verify_release.py
```

This command checks the FASTA records, 21-dimensional labels, sample counts, feature dimensions, finite values, and merged feature matrices. A successful check ends with:

```text
PUBLIC RELEASE CHECK: PASS
```

### 2. Parse peptide sequences and labels

The FASTA header of each peptide contains a 21-bit binary label vector. Convert the training and test FASTA files to tabular CSV files with:

```bash
python scripts/read_fasta_labels.py
```

The generated files are:

```text
parsed_labels/train_sequences_labels.csv
parsed_labels/test_sequences_labels.csv
```

Each row contains a sample identifier, peptide sequence, sequence length, number of positive labels, and 21 individual label columns. A different output directory can be specified as follows:

```bash
python scripts/read_fasta_labels.py --out_dir outputs/parsed_labels
```

### 3. Load the prepared 3,387-dimensional features

The concatenated feature matrices can be loaded directly:

```python
import numpy as np

X_train = np.load("features/merged/train_features_3387.npy")
X_test = np.load("features/merged/test_features_3387.npy")

print(X_train.shape)  # (7872, 3387)
print(X_test.shape)   # (1969, 3387)
```

### 4. Reconstruct the merged features

To load the eight individual feature groups, apply the required dimensional expansion, and concatenate them in the predefined order, run:

```bash
python scripts/load_features_demo.py
```

By default, the script writes:

```text
features/merged/train_features_3387.npy
features/merged/test_features_3387.npy
```

To preserve the supplied matrices and write reconstructed files elsewhere:

```bash
python scripts/load_features_demo.py --out_dir outputs/merged_features
```

### 5. Extract pretrained sequence representations

The supplied short-peptide encoder is a BERT masked-language model pretrained on UniRef90 peptide sequences. The following example encodes eight test peptides and mean-pools their residue representations:

```bash
python scripts/extract_pretrained_features_demo.py --split test --limit 8 --output outputs/test_pretrained_features.npy
```

The output is a NumPy array with shape `(8, 128)`. The split, number of sequences, batch size, and output path can be changed through command-line arguments:

```bash
python scripts/extract_pretrained_features_demo.py --split train --limit 100 --batch_size 32 --output outputs/train_pretrained_features.npy
```

### 6. Evaluate multi-label predictions

Prepare the ground-truth labels and prediction scores as two-dimensional `.npy` arrays or headerless `.csv` matrices with the same shape, normally `(number_of_samples, 21)`. Then run:

```bash
python scripts/evaluate_predictions.py --true path/to/true_labels.npy --pred path/to/predicted_scores.npy --threshold 0.57
```

The script reports the five sample-based multi-label measures used in this project:

- Accuracy;
- Precision;
- Coverage;
- Absolute true;
- Absolute false.

If the prediction matrix already contains binary values, the threshold is not applied. Results can also be saved as JSON:

```bash
python scripts/evaluate_predictions.py --true path/to/true_labels.npy --pred path/to/predicted_scores.npy --threshold 0.57 --output outputs/metrics.json
```

## Dataset

The benchmark contains 7,872 training peptides and 1,969 independent test peptides. Peptide lengths range from 1 to 50 amino-acid residues. A FASTA record is formatted as follows:

```text
>000000000000000010000
ARRRRCSDRFRNCPADEALCGRRRR
```

The fixed label order is:

```text
AAP, ABP, ACP, ACVP, ADP, AEP, AFP, AHIVP, AHP, AIP, AMRSAP,
APP, ATP, AVP, BBP, BIP, CPP, DPPIP, QSP, SBP, THP
```

See [`dataset/LABELS.md`](dataset/LABELS.md) for the indexed label list.

## Feature representation

Eight feature groups are supplied. After preprocessing, they form a 3,387-dimensional representation for each peptide.

| Feature group | Stored format | Original dimension | Input dimension |
|---|---:|---:|---:|
| AAC | CSV | 20 | 400 |
| DDE | CSV | 400 | 400 |
| CKSAAGP | CSV | 100 | 400 |
| PAAC | CSV | 22 | 396 |
| APAAC | CSV | 24 | 384 |
| AAindex1 | NPY | 566 | 566 |
| CTD | NPY | 441 | 441 |
| MorganFP | NPY | 400 | 400 |
| **Total** |  |  | **3,387** |

The fixed concatenation order and column intervals are:

```text
AAC:         0-400
DDE:       400-800
CKSAAGP:   800-1200
PAAC:     1200-1596
APAAC:    1596-1980
AAindex1: 1980-2546
CTD:      2546-2987
MorganFP: 2987-3387
```

The Morgan fingerprints were generated as 1,024-bit radius-2 representations and reduced to 400 dimensions using a scaler and PCA fitted on the training set. The fitted preprocessing objects and parameter description are included under `features/MorganFP/`.

## Pretrained peptide encoder

The `pre_trained_model/` directory contains the model configuration, vocabulary, tokenizer files, and weights required by `extract_pretrained_features_demo.py`. Its principal settings are:

| Setting | Value |
|---|---:|
| Architecture | BertForMaskedLM |
| Pretraining task | Masked language modeling |
| Vocabulary size | 26 |
| Hidden size | 128 |
| Hidden layers | 6 |
| Attention heads | 8 |
| Maximum positions | 128 |

Additional information is recorded in `pre_trained_model/pretraining_metadata.json`.

## Utility scripts

| Script | Purpose |
|---|---|
| `read_fasta_labels.py` | Parse peptide sequences and 21-bit labels from FASTA files. |
| `load_features_demo.py` | Load, validate, expand, and concatenate the eight feature groups. |
| `extract_pretrained_features_demo.py` | Extract 128-dimensional representations with the supplied peptide encoder. |
| `evaluate_predictions.py` | Calculate five multi-label evaluation measures. |
| `verify_release.py` | Check data integrity and feature dimensions. |

## Notes

- Keep the original sample order unchanged when aligning sequences, labels, and feature matrices.
- Run the scripts from the repository root unless custom paths are explicitly supplied.
- If a downloaded `.npy`, `.pkl`, or `.safetensors` file appears to be a small text pointer, run `git lfs pull` before using it.
- Do not add CSV index columns or headers to the supplied feature matrices.

## Citation

If you use the dataset, feature representations, pretrained encoder, or utility scripts in your research, please cite the accompanying PSMF-MTP manuscript.

## Contact

For questions about this repository, please open an issue at [MuJianhang/PSMF-MTP](https://github.com/MuJianhang/PSMF-MTP/issues).
