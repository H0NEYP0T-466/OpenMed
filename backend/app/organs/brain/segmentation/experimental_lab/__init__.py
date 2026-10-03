"""Experimental Laboratory — LiteMedSAM promptable segmentation.

Everything specific to the LiteMedSAM segmentation model lives here: the
architecture, the prompt geometry, the serving pipeline, the checkpoint
handling and the Kaggle fine-tuning package.

It sits apart from its parent package on purpose. LiteMedSAM is the current
experimental segmentation engine, not a settled part of the product: the
automated heatmap-prompt path was retired from the classifier run, and a
replacement model is expected to be trained. Keeping the whole stack in one
folder means the rest of the brain organ stays free of a model that is on its
way out.

The API surface is unchanged — ``/api/brain/segment``,
``/api/brain/segment-click`` and ``/api/brain/segment-box`` are still served
from the same URLs by :mod:`.router`, which is mounted by ``app.main``.

Note on the two prompt paths:

* The **assistive** path (click, drawn box) is the one in active use. A click
  becomes a 48px box centred on it; a drawn box is used as-is. Both measured
  well against ground truth and both skip classification entirely.
* The **automated** path (Grad-CAM heatmap → box) is retained for the API but
  is no longer wired into the workspace, because the CAM box scored 0.31 mean
  Dice against ground truth where a clinician-supplied box reaches 0.92.
"""
