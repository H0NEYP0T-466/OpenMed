"""
3D brain region mapping for classification → localization.

Maps the 38 anatomical locations from the brain tumor dataset to approximate
3D coordinates in MNI-like space. Used by the API to tell the frontend where
to highlight on the 3D brain model.

Coordinates are approximate centroids in a normalized [-1, 1]³ brain space
oriented as: X=right, Y=anterior, Z=superior.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Complete region registry (all 38 locations from the dataset) ──────────

BRAIN_REGIONS_3D: dict[str, dict] = {
    # Major lobes
    "frontal": {
        "name": "frontal",
        "display_name": "Frontal Lobe",
        "coordinates_3d": [0.0, 0.65, 0.30],
        "color": "#ee7c6a",
        "lobe": "Frontal",
        "description": "Anterior cerebral cortex — executive function, motor planning, Broca's area.",
    },
    "temporal": {
        "name": "temporal",
        "display_name": "Temporal Lobe",
        "coordinates_3d": [0.70, -0.10, -0.20],
        "color": "#6393d8",
        "lobe": "Temporal",
        "description": "Lateral cortex — auditory processing, hippocampus, memory consolidation.",
    },
    "parietal": {
        "name": "parietal",
        "display_name": "Parietal Lobe",
        "coordinates_3d": [0.0, -0.20, 0.70],
        "color": "#f2a33b",
        "lobe": "Parietal",
        "description": "Superior posterior cortex — somatosensory integration, spatial orientation.",
    },
    "occipital": {
        "name": "occipital",
        "display_name": "Occipital Lobe",
        "coordinates_3d": [0.0, -0.80, 0.10],
        "color": "#d89bc4",
        "lobe": "Occipital",
        "description": "Posterior cortex — primary and association visual processing.",
    },
    # Deep structures
    "ventricle": {
        "name": "ventricle",
        "display_name": "Ventricular System",
        "coordinates_3d": [0.0, 0.0, 0.10],
        "color": "#33cccc",
        "lobe": "Internal",
        "description": "CSF-filled cavities (lateral, third, fourth ventricles).",
    },
    "intraventricular": {
        "name": "intraventricular",
        "display_name": "Intraventricular",
        "coordinates_3d": [0.0, 0.05, 0.12],
        "color": "#2eb8b8",
        "lobe": "Internal",
        "description": "Within the ventricle lumen — choroid plexus tumors, colloid cysts.",
    },
    "basal ganglia": {
        "name": "basal ganglia",
        "display_name": "Basal Ganglia",
        "coordinates_3d": [0.25, 0.05, 0.05],
        "color": "#e67e22",
        "lobe": "Subcortical",
        "description": "Deep grey nuclei — caudate, putamen, globus pallidus.",
    },
    "thalamus": {
        "name": "thalamus",
        "display_name": "Thalamus",
        "coordinates_3d": [0.0, -0.10, 0.05],
        "color": "#c0392b",
        "lobe": "Diencephalon",
        "description": "Central relay station for sensory and motor signals to the cortex.",
    },
    "hypothalamus": {
        "name": "hypothalamus",
        "display_name": "Hypothalamus",
        "coordinates_3d": [0.0, 0.05, -0.10],
        "color": "#e74c3c",
        "lobe": "Diencephalon",
        "description": "Autonomic and endocrine regulation — pituitary gland control.",
    },
    # Posterior fossa
    "brainstem": {
        "name": "brainstem",
        "display_name": "Brainstem",
        "coordinates_3d": [0.0, -0.20, -0.55],
        "color": "#8e44ad",
        "lobe": "Brainstem",
        "description": "Midbrain, pons, and medulla — vital autonomic centres.",
    },
    "cerebellopontine angle": {
        "name": "cerebellopontine angle",
        "display_name": "Cerebellopontine Angle (CPA)",
        "coordinates_3d": [0.55, -0.45, -0.40],
        "color": "#3498db",
        "lobe": "Posterior Fossa",
        "description": "Junction of cerebellum, pons, and petrous temporal bone — schwannoma hotspot.",
    },
    "cerebellum": {
        "name": "cerebellum",
        "display_name": "Cerebellum",
        "coordinates_3d": [0.0, -0.70, -0.45],
        "color": "#1abc9c",
        "lobe": "Posterior Fossa",
        "description": "Motor coordination, procedural learning, balance.",
    },
    "posterior fossa": {
        "name": "posterior fossa",
        "display_name": "Posterior Fossa",
        "coordinates_3d": [0.0, -0.60, -0.35],
        "color": "#2980b9",
        "lobe": "Posterior Fossa",
        "description": "Infratentorial compartment containing cerebellum and brainstem.",
    },
    # Midline & falx
    "falx": {
        "name": "falx",
        "display_name": "Falx Cerebri",
        "coordinates_3d": [0.0, 0.0, 0.50],
        "color": "#c58696",
        "lobe": "Meningeal",
        "description": "Midline dural fold separating cerebral hemispheres.",
    },
    "parafalcine": {
        "name": "parafalcine",
        "display_name": "Parafalcine",
        "coordinates_3d": [0.08, -0.05, 0.55],
        "color": "#b57585",
        "lobe": "Meningeal",
        "description": "Adjacent to falx cerebri — common meningioma site.",
    },
    "tentorial": {
        "name": "tentorial",
        "display_name": "Tentorium Cerebelli",
        "coordinates_3d": [0.0, -0.40, -0.10],
        "color": "#a76f7a",
        "lobe": "Meningeal",
        "description": "Dural fold separating cerebrum from cerebellum.",
    },
    # Sellar / suprasellar
    "sella turcica": {
        "name": "sella turcica",
        "display_name": "Sella Turcica",
        "coordinates_3d": [0.0, 0.10, -0.25],
        "color": "#f39c12",
        "lobe": "Sellar",
        "description": "Bony saddle housing the pituitary gland.",
    },
    "sellar": {
        "name": "sellar",
        "display_name": "Sellar Region (Pituitary Fossa)",
        "coordinates_3d": [0.0, 0.10, -0.25],
        "color": "#f39c12",
        "lobe": "Sellar",
        "description": "Hypophyseal fossa housing the pituitary gland and stalk.",
    },
    "pituitary": {
        "name": "pituitary",
        "display_name": "Pituitary Gland / Sella",
        "coordinates_3d": [0.0, 0.10, -0.25],
        "color": "#f39c12",
        "lobe": "Sellar",
        "description": "Pituitary gland (hypophysis) located within sella turcica.",
    },
    "suprasellar": {
        "name": "suprasellar",
        "display_name": "Suprasellar",
        "coordinates_3d": [0.0, 0.12, -0.15],
        "color": "#d4a017",
        "lobe": "Sellar",
        "description": "Above the sella turcica — optic chiasm, hypothalamus.",
    },
    "sellar-suprasellar": {
        "name": "sellar-suprasellar",
        "display_name": "Sellar–Suprasellar",
        "coordinates_3d": [0.0, 0.11, -0.20],
        "color": "#e8b930",
        "lobe": "Sellar",
        "description": "Combined sellar and suprasellar region.",
    },
    # Cranial fossa
    "anterior cranial fossa": {
        "name": "anterior cranial fossa",
        "display_name": "Anterior Cranial Fossa",
        "coordinates_3d": [0.0, 0.60, -0.10],
        "color": "#e84393",
        "lobe": "Skull Base",
        "description": "Frontal skull base housing frontal lobes.",
    },
    "middle cranial fossa": {
        "name": "middle cranial fossa",
        "display_name": "Middle Cranial Fossa",
        "coordinates_3d": [0.50, 0.10, -0.25],
        "color": "#d63031",
        "lobe": "Skull Base",
        "description": "Lateral skull base housing temporal lobes and sella.",
    },
    "posterior cranial fossa": {
        "name": "posterior cranial fossa",
        "display_name": "Posterior Cranial Fossa",
        "coordinates_3d": [0.0, -0.55, -0.40],
        "color": "#6c5ce7",
        "lobe": "Skull Base",
        "description": "Posterior skull base containing cerebellum and brainstem.",
    },
    # Pineal & third ventricle
    "pineal": {
        "name": "pineal",
        "display_name": "Pineal Region",
        "coordinates_3d": [0.0, -0.30, 0.15],
        "color": "#00b894",
        "lobe": "Pineal",
        "description": "Posterior to third ventricle — germinoma, pineocytoma.",
    },
    "third ventricle": {
        "name": "third ventricle",
        "display_name": "Third Ventricle",
        "coordinates_3d": [0.0, -0.05, 0.05],
        "color": "#00cec9",
        "lobe": "Internal",
        "description": "Midline CSF cavity between the thalami.",
    },
    "fourth ventricle": {
        "name": "fourth ventricle",
        "display_name": "Fourth Ventricle",
        "coordinates_3d": [0.0, -0.45, -0.35],
        "color": "#81ecec",
        "lobe": "Posterior Fossa",
        "description": "CSF cavity between brainstem and cerebellum.",
    },
    # Trigeminal / CN
    "trigeminal": {
        "name": "trigeminal",
        "display_name": "Trigeminal Nerve (CN V)",
        "coordinates_3d": [0.45, -0.15, -0.30],
        "color": "#fd79a8",
        "lobe": "Cranial Nerve",
        "description": "Fifth cranial nerve — schwannoma site.",
    },
    # Supratentorial general
    "supratentorial": {
        "name": "supratentorial",
        "display_name": "Supratentorial",
        "coordinates_3d": [0.0, 0.0, 0.35],
        "color": "#a29bfe",
        "lobe": "Supratentorial",
        "description": "Above the tentorium — cerebral hemispheres and deep nuclei.",
    },
    # White matter & tracts
    "centrum semiovale": {
        "name": "centrum semiovale",
        "display_name": "Centrum Semiovale",
        "coordinates_3d": [0.20, 0.0, 0.50],
        "color": "#dfe6e9",
        "lobe": "White Matter",
        "description": "Deep white matter mass of the cerebral hemispheres.",
    },
    "corpus callosum": {
        "name": "corpus callosum",
        "display_name": "Corpus Callosum",
        "coordinates_3d": [0.0, 0.0, 0.30],
        "color": "#b2bec3",
        "lobe": "White Matter",
        "description": "Major commissure connecting left and right hemispheres.",
    },
    "internal capsule": {
        "name": "internal capsule",
        "display_name": "Internal Capsule",
        "coordinates_3d": [0.25, -0.05, 0.10],
        "color": "#636e72",
        "lobe": "White Matter",
        "description": "White matter tract between basal ganglia and thalamus.",
    },
    # Cavernous sinus
    "cavernous sinus": {
        "name": "cavernous sinus",
        "display_name": "Cavernous Sinus",
        "coordinates_3d": [0.35, 0.15, -0.20],
        "color": "#e17055",
        "lobe": "Vascular",
        "description": "Dural venous sinus lateral to the sella — meningioma, schwannoma.",
    },
    # Insular / sylvian
    "insular": {
        "name": "insular",
        "display_name": "Insular Cortex",
        "coordinates_3d": [0.55, 0.05, 0.0],
        "color": "#fab1a0",
        "lobe": "Insular",
        "description": "Deep cortex beneath the Sylvian fissure — glioma hotspot.",
    },
    "sylvian fissure": {
        "name": "sylvian fissure",
        "display_name": "Sylvian Fissure",
        "coordinates_3d": [0.55, 0.10, -0.05],
        "color": "#fdcb6e",
        "lobe": "Fissure",
        "description": "Lateral fissure separating frontal/parietal from temporal lobe.",
    },
    # Sphenoid / petrosal
    "sphenoid wing": {
        "name": "sphenoid wing",
        "display_name": "Sphenoid Wing",
        "coordinates_3d": [0.50, 0.30, -0.15],
        "color": "#ffeaa7",
        "lobe": "Skull Base",
        "description": "Lateral sphenoid bone — meningioma attachment site.",
    },
    "petrous": {
        "name": "petrous",
        "display_name": "Petrous Bone",
        "coordinates_3d": [0.60, -0.20, -0.35],
        "color": "#dfe6e9",
        "lobe": "Skull Base",
        "description": "Dense temporal bone housing the inner ear and facial nerve canal.",
    },
    # Optic / chiasmatic
    "optic nerve": {
        "name": "optic nerve",
        "display_name": "Optic Nerve / Chiasm",
        "coordinates_3d": [0.0, 0.25, -0.10],
        "color": "#74b9ff",
        "lobe": "Cranial Nerve",
        "description": "CN II — optic pathway glioma site.",
    },
    # Foramen magnum
    "foramen magnum": {
        "name": "foramen magnum",
        "display_name": "Foramen Magnum",
        "coordinates_3d": [0.0, -0.35, -0.65],
        "color": "#a29bfe",
        "lobe": "Craniocervical",
        "description": "Opening at skull base where brainstem becomes spinal cord.",
    },
    # Jugular foramen
    "jugular foramen": {
        "name": "jugular foramen",
        "display_name": "Jugular Foramen",
        "coordinates_3d": [0.45, -0.30, -0.50],
        "color": "#636e72",
        "lobe": "Skull Base",
        "description": "Transmits CN IX/X/XI and internal jugular vein.",
    },
    # Convexity (generic dural)
    "convexity": {
        "name": "convexity",
        "display_name": "Cerebral Convexity",
        "coordinates_3d": [0.40, 0.0, 0.55],
        "color": "#c58696",
        "lobe": "Meningeal",
        "description": "Outer brain surface beneath the calvarium — meningioma site.",
    },
}


# ── Tumor-type → typical location fallback ────────────────────────────────

_TUMOR_LOCATION_FALLBACK: dict[str, list[str]] = {
    "Meningioma": ["falx", "convexity", "sphenoid wing", "parafalcine"],
    "Astrocytoma": ["frontal", "temporal", "insular", "parietal"],
    "Glioblastoma": ["frontal", "temporal", "parietal", "corpus callosum"],
    "Schwannoma": ["cerebellopontine angle", "trigeminal"],
    "Normal": [],
    "Ependymoma - Subependymoma": ["ventricle", "fourth ventricle", "posterior fossa"],
    "Medulloblastoma": ["posterior fossa", "cerebellum", "fourth ventricle"],
    "Oligodendroglioma": ["frontal", "parietal", "centrum semiovale"],
    "Pituitary": ["sellar", "sella turcica", "suprasellar"],
    "Hemangiopericytoma": ["falx", "tentorial", "convexity"],
    "Neurocytoma": ["ventricle", "intraventricular"],
    "Dysembryoplastic Neuroepithelial Tumor": ["temporal", "frontal"],
    "Germinoma": ["pineal", "suprasellar", "basal ganglia"],
    "Ganglioglioma": ["temporal", "frontal"],
}


def map_prediction_to_3d(
    predicted_class: str,
    locations: Optional[list[str]] = None,
) -> list[dict]:
    """
    Map a classification result + optional location metadata to 3D brain regions.

    Parameters
    ----------
    predicted_class : str
        Full class name, e.g. ``"Meningioma T1C+"``.
    locations : list[str] | None
        Anatomical location tags from the image metadata.  When absent the
        function falls back to typical locations for the predicted tumor type.

    Returns
    -------
    list[dict]
        Region dicts with an added ``probability`` key.
    """
    # Parse tumor type from class name
    tumor_type = predicted_class.rsplit(" ", 1)[0] if " " in predicted_class else predicted_class

    if not locations:
        locations = _TUMOR_LOCATION_FALLBACK.get(tumor_type, ["supratentorial"])

    mapped: list[dict] = []
    for i, loc in enumerate(locations):
        loc_lower = loc.lower().strip()
        region = BRAIN_REGIONS_3D.get(loc_lower)
        if region:
            entry = dict(region)
            # Assign decreasing probability — first location is most likely
            entry["probability"] = round(max(0.90 - i * 0.12, 0.25), 2)
            mapped.append(entry)
        else:
            logger.debug(f"Unknown brain region '{loc}' — skipping 3D mapping")

    return mapped
