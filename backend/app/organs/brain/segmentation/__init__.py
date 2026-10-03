"""Brain organ — segmentation.

The LiteMedSAM segmentation stack lives in :mod:`.experimental_lab`, kept
separate because that engine is being replaced: the automated heatmap-prompt
path has been retired from the classifier run and a retrained model is
expected.

Nothing in this package is imported directly by the application any more —
``app.main`` mounts the experimental lab's router, and the classifier run no
longer segments. This namespace stays as the home for whichever segmentation
model the brain organ uses next.
"""
