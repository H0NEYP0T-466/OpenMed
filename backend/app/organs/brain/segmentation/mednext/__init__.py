"""Brain segmentation — MedNeXt-2D.

Automated lesion segmentation that runs after the EfficientNetV2-B2 classifier
whenever the predicted class is not ``Normal``. Unlike the LiteMedSAM
assistive path in :mod:`..experimental_lab`, it needs no prompt: the network
reads the slice and outputs the mask.

The package is deliberately light to import (no FastAPI, no training code) so
the Kaggle training entry point can put this directory on ``sys.path`` and use
``model`` / ``preprocessor`` / ``pipeline`` without pulling in the application.

Modules
-------
model         architecture, spec, UpKern transfer (shared by training + serving)
preprocessor  the train/serve image contract (shared by training + serving)
pipeline      checkpoint loading and single-image inference
router        FastAPI endpoints (imported by ``app.main`` only)
training/     Kaggle training package, driven by ``train_mednext.py``
"""
