# Project Dustbin (Brain Classification Archive)

This directory archives deprecated training runs, weights, visuals, and logs from prior experimental iterations.

### Archives:
- `checkpoints/`: Model I initial checkpoint (`brain_best_model.pth` from Sep 18, 39 classes).
- `visuals/`: Model I 39-class training curves, metrics, confusion matrix, and prediction samples.
- `public_visuals/`: Model I public assets served to the frontend.
- `legacy_42class/`: Model II 42-class de-leaking run artifacts (Sep 19):
  - `checkpoints/`: 42-class model checkpoint and label space.
  - `visuals/`: 42-class confusion matrix, ROC/PR curves, classification report, and training log.
  - `public_visuals/`: 42-class public web assets.
- `kaggle_training_scripts.zip`: Previous training runner script bundle.

*Note: This folder is excluded from version control via `.gitignore`.*
