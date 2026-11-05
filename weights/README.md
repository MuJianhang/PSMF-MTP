# Trained PSMF-MTP weights

The file `psmf_mtp_100epochs_seed205808.pt` is the 100-epoch, seed-205808, complete-feature PSMF-MTP fusion/classifier state dictionary. It contains 6,877,163 parameter values. The frozen sequence encoder is supplied separately in `../pre_trained_model/`.

- Settings and SHA-256: [model_config.json](model_config.json)
- Individual run record: [training_record.json](training_record.json)
- SHA-256: `c70c0420428b6e7291a1a2cbe048525ab032e49acb02508f2521368fef82e1e3`

A matching local architecture definition is required to use a `state_dict` for prediction. The core implementation is retained locally; this parameter file contains no training script or custom loss source. Single-checkpoint metrics are distinct from repeated-run summaries.
