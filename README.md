# PSMF-MTP

[English](README.md) | [简体中文](README.zh-CN.md)

PSMF-MTP predicts multiple therapeutic functions for short peptides by combining a pretrained sequence encoder, eight physicochemical and structural feature groups, Mamba-based fusion, and a Transformer classifier.

This guide describes the overall project workflow, the released weights and data, preparation of a compatible dataset, and the local PyCharm demonstration.

## Public resources and implementation scope

| Resource | Location |
|---|---|
| Benchmark sequences and 21-label annotations | `dataset/` |
| Eight feature groups and merged 3,387-dimensional matrices | `features/` |
| Frozen short-peptide sequence encoder weights and tokenizer | `pre_trained_model/` |
| Trained 100-epoch fusion/classification weights | `weights/psmf_mtp_100epochs_seed205808.pt` |
| Checkpoint settings and individual training record | `weights/model_config.json`, `weights/training_record.json` |
| Small aligned input example | `examples/small_demo/` |
| Data preparation, sequence extraction, weight inspection, and evaluation utilities | `scripts/` |
| PyCharm recording guide and future video location | [videos/README.md](videos/README.md) |

The core architecture implementation, custom loss implementation, and complete training/inference entry points are retained locally at this stage. The public utilities can prepare inputs, verify weights, and evaluate saved predictions. Complete prediction in the planned video uses the local project with its retained core implementation; downloading a state dictionary does not by itself provide that implementation.

## Overall workflow

1. Prepare peptide sequences and functional annotations, and keep training/test data separate.
2. Encode sequences with the frozen UniRef90 short-peptide encoder.
3. Prepare AAC, DDE, CKSAAGP, PAAC, APAAC, AAindex1, CTD, and MorganFP features, and combine them in the fixed order.
4. Map the sequence and descriptor features into a common representation space and fuse them through Mamba.
5. Apply the Transformer classifier and sigmoid output to obtain 21 functional-label probabilities.
6. During training, optimize the mapping/fusion/classification components while keeping the encoder frozen, and save the trained weights.
7. During inference, load the corresponding encoder and predictor weights, then apply the configured threshold to obtain predicted label sets.
8. When true labels are available, evaluate prediction quality and inspect label-wise results.

The manuscript also examines comparison methods, feature and component ablations, descriptor dimensionality, downstream transfer of the encoder, auxiliary label-context analysis, label-wise performance, and representation/feature-group interpretation.

## Environment and download

The public utilities require Python 3.10 or later. Download large files using Git LFS:

```
git lfs install
git clone https://github.com/MuJianhang/PSMF-MTP.git
cd PSMF-MTP
git lfs pull
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```
source .venv/bin/activate
```

Install the public-tool dependencies:

```
python -m pip install -r requirements.txt
```

The preprocessing and weight-inspection utilities can run on CPU. The complete local model environment additionally needs the retained implementation and its Mamba dependencies; the manuscript used Ubuntu under WSL with CUDA. In PyCharm, choose the interpreter that actually has those dependencies installed.

## Released trained checkpoint

The released checkpoint contains 6,877,163 parameter values for the sequence mapping, eight descriptor mappings, Mamba fusion, Transformer classifier, and output head. The frozen encoder is stored separately in `pre_trained_model/`; both components are needed for full prediction.

| Setting | Selected checkpoint |
|---|---|
| Feature configuration | Complete 3,387-dimensional descriptor set |
| Training epochs / seed | 100 / 205808 |
| Batch size / optimizer | 64 / AdamW |
| Learning rate / weight decay | `8e-5` / `0.15` |
| Dropout | `0.4` |
| Fusion dimension / tokens | 80 / 41 |
| Mamba | One block; state 16; convolution 4; expansion 4 |
| Transformer | One layer; four heads; feedforward dimension 256 |
| Output labels / threshold | 21 / `0.57` |

This is one individual checkpoint from the 100-epoch full-feature configuration. Its training record describes that run; its metrics are not a five-run mean. Weights are supplied as a PyTorch `state_dict` and require the matching model definition for prediction. Configuration and SHA-256 checksums are recorded in [weights/model_config.json](weights/model_config.json).

Check the release and inspect the actual weights:

```
python scripts/verify_release.py
python scripts/inspect_model_weights.py
```

Successful checks end with `PUBLIC RELEASE CHECK: PASS` and `PSMF-MTP WEIGHTS CHECK: PASS`.

## Example inputs

The small example contains 128 training and 32 test records, with matching descriptor matrices:

```
examples/small_demo/
|-- dataset/
|   |-- train_new.fasta
|   `-- test_new.fasta
|-- features/merged/
|   |-- train_features_3387.npy
|   `-- test_features_3387.npy
`-- provenance.json
```

Parse annotations and export label matrices:

```
python scripts/read_fasta_labels.py --dataset_dir examples/small_demo/dataset --out_dir outputs/small_demo/labels --save_npy
```

This saves sequence/label CSVs, `train_labels.npy`, and `test_labels.npy`. To generate the encoder inputs used by the selected checkpoint:

```
python scripts/extract_sequence_features.py --fasta examples/small_demo/dataset/test_new.fasta --output outputs/small_demo/test_sequence_features.npy --device cpu
```

The sequence matrix has shape `(32, 6400)`; the supplied test descriptor matrix has shape `(32, 3387)`. Both retain the original FASTA record order.

## Use your own dataset

### 1. Prepare sequences and labels

Use the same 21 functional definitions and label order. Ordinary FASTA identifiers are sufficient for sequence extraction and prediction; known labels are required for training/evaluation. A labeled header can contain an identifier followed by 21 binary values:

```
>peptide_001 labels=010000100000000000000
ARRRRCSDRFRNCPADEALCGRRRR
```

This record illustrates syntax. Actual labels must come from your annotations. The second and seventh bits denote ABP and AFP.

Keep prepared files separate from the supplied benchmark:

```
data/my_dataset/
|-- dataset/
|   |-- train_new.fasta
|   `-- test_new.fasta
`-- features/merged/
    |-- train_features_3387.npy
    `-- test_features_3387.npy
```

```
python scripts/read_fasta_labels.py --dataset_dir data/my_dataset/dataset --out_dir outputs/my_dataset/labels --save_npy
```

The checkpoint has a fixed 21-label output. Different target definitions require corresponding model/output adaptation and retraining.

### 2. Prepare sequence representations

```
python scripts/extract_sequence_features.py --fasta data/my_dataset/dataset/test_new.fasta --output outputs/my_dataset/test_sequence_features.npy --device cpu
```

Repeat with the training FASTA and a different output path when preparing training inputs. Unlabeled FASTA files can be processed using the same command. Use `--max_samples` for a small initial check.

For compatibility with the selected 100-epoch checkpoint, the helper uses 50 token positions including `[CLS]` and `[SEP]`, retaining up to the first 48 residues from each 1-50-residue input. It flattens all 50 hidden-state positions into 6,400 values and reproduces the training cache's float16 rounding before exporting float32 inputs. The accompanying `.ids.csv` and `.metadata.json` record sample order and preprocessing details.

### 3. Prepare descriptor features

Each split needs a finite matrix of shape `(number_of_samples, 3387)` with exactly the same row order as its sequences. Use the feature definitions and settings in the manuscript.

If the eight feature-group files are already prepared, combine them with:

```
python scripts/load_features_demo.py --feature_root data/my_dataset/features --out_dir data/my_dataset/features/merged
```

| Group | Training filename | Stored columns | Model columns |
|---|---|---:|---:|
| AAC | `AAC/train_aac.csv` | 20 | 400 |
| DDE | `DDE/train_dde.csv` | 400 | 400 |
| CKSAAGP | `CKSAAGP/train_cksaagp.csv` | 100 | 400 |
| PAAC | `PAAC/train_paac.csv` | 22 | 396 |
| APAAC | `APAAC/train_apaac.csv` | 24 | 384 |
| AAindex1 | `AAindex1/train_aaindex1.npy` | 566 | 566 |
| CTD | `CTD/train_ctd.npy` | 441 | 441 |
| MorganFP | `MorganFP/train_morganfp_400.npy` | 400 | 400 |

Use the `test` prefix for test files. CSV feature files must have no header or index column. Concatenation follows the table order. The loader merges existing group files; complete extraction of all eight groups from arbitrary raw FASTA is not included in the public utilities.

For inference with the released checkpoint, preserve the original feature definitions/scales and reuse the training-fitted Morgan preprocessing objects in `features/MorganFP/`. This checkpoint does not apply an additional global descriptor standardization. If developing a newly trained model, fit any data-dependent preprocessing using training data only.

### 4. Run the complete model in your local project

Open the full local PSMF-MTP project in PyCharm, select its configured interpreter, and set the working directory to that project root. In your retained inference entry point, specify the input FASTA, aligned 6,400-dimensional sequence representations and 3,387-dimensional descriptors, pretrained encoder location, released predictor checkpoint, threshold `0.57`, and output directory. Load the weights into the matching architecture, switch to evaluation mode, and run prediction.

The core entry point is retained locally and is not supplied as a public script in this release. The planned video will show the actual run configuration and execution. To use another compatible dataset, replace both the sequence/annotation paths and the matching feature paths together; loading an unrelated feature matrix of the same shape is not sufficient.

## Outputs and evaluation

The local inference stage should export a probability matrix of shape `(number_of_samples, 21)`, thresholded predictions, and a sample/sequence mapping. Preserve this row order when evaluating. A CSV table with sequence identifiers, predicted labels, and probabilities is suitable for inspecting the results in the video.

For saved scores and matching true labels, replace the example prediction path below with the actual local output:

```
python scripts/evaluate_predictions.py --true outputs/small_demo/labels/test_labels.npy --pred path/to/predicted_scores.npy --threshold 0.57 --output outputs/small_demo/metrics.json
```

The evaluator accepts `.npy` or headerless numerical CSV matrices. It reports Accuracy (sample-wise Jaccard agreement), Precision, Coverage (sample-wise recall), Absolute true (exact label-set match), and Absolute false (Hamming loss). True labels are needed to calculate metrics, while prediction of unlabeled peptides remains possible. Binary prediction matrices are evaluated directly.

The fixed label order is:

```
AAP, ABP, ACP, ACVP, ADP, AEP, AFP, AHIVP, AHP, AIP, AMRSAP,
APP, ATP, AVP, BBP, BIP, CPP, DPPIP, QSP, SBP, THP
```

See [dataset/LABELS.md](dataset/LABELS.md). The full benchmark has 7,872 training and 1,969 test peptides; source indices for the small example are in `provenance.json`.

## Local PyCharm video

The recording is planned for approximately 1-2 minutes: select example/own-data paths, show the local configuration, load the trained weights, run prediction, inspect outputs, and show where to substitute another dataset.

The video has not yet been recorded. Recording guidance and the intended upload directory are provided in [videos/README.md](videos/README.md). Add the actual video link after uploading the recording.

## Citation and contact

Please cite the accompanying PSMF-MTP manuscript when using these resources. Questions can be submitted through [GitHub issues](https://github.com/MuJianhang/PSMF-MTP/issues).
