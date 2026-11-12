# Lung dataset hygiene report

- Generated: `2026-11-11T11:38:33+00:00`
- Dataset root: `/kaggle/input/datasets/tawsifurrahman/covid19-radiography-database/COVID-19_Radiography_Dataset`
- Images: **20858** (20858 readable)
- pHash: `imagehash`, Hamming ≤ 5
- **Verdict: `WARN`**

## Class counts

| Class | Images |
|---|---|
| COVID | 3365 |
| Lung_Opacity | 5889 |
| Normal | 10278 |
| Viral Pneumonia | 1326 |

Imbalance ratio: **7.7511**

## Duplicates

- Exact duplicates (SHA-256): 0 group(s), 0 file(s)
- Near duplicates (pHash): 518 cluster(s), 1853 file(s), 1618 pair(s)
- Clusters total: 19523 (largest 390)

## Cross-class conflicts

None.

## Unreadable files

None.

## Size and mode histogram

| Size | Count |
|---|---|
| 299x299 | 20858 |

Modes: Lx20718, RGBx140

## Leakage audit

- Clusters spanning splits: **0** (must be 0)
- Split sizes: {'train': 14598, 'val': 3130, 'test': 3130}
- Source pools in more than one split: 3

| Source pool | Splits |
|---|---|
| covid_mixed | test, train, val |
| rsna | test, train, val |
| viral | test, train, val |

## Verdict

`WARN`

- WARN: 518 near-duplicate cluster(s) were grouped
- WARN: class imbalance 7.75:1


## Conflict resolution

- Exact duplicates dropped: **54**
- Cross-class conflicts relabeled (majority): **236**
- Cross-class conflict ties dropped: **253**
- Single-label near-duplicates collapsed: **0**
- Kept images: **20858**
- New imbalance ratio: **7.7511:1**
