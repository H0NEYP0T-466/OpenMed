#!/usr/bin/env python3
"""OpenMed SRS — Database Diagram (Figure 5.6.10).

The physical schema behind the ER model: table names, column names, SQL types,
key constraints and the indexes the query patterns need. Release 1.0 targets
SQLite for the single-clinic deployment and the same DDL runs on PostgreSQL
unchanged apart from BIGSERIAL.

Run: python3 database_diagram.py
Out: database_openmed.svg
"""

from svgkit import Canvas, build

W, H = 1500, 1040
XS = [60, 560, 1060]
BW = 380
ROWS = [150, 430, 690]

c = Canvas(
    W, H,
    "OpenMed — Database Diagram",
    subtitle="Physical schema, release 1.0 · PK primary key · FK foreign key "
             "· UQ unique",
    footer="Append-only by policy: report_version never loses a row and the "
           "application role holds no UPDATE or DELETE grant on audit_log.",
)

F_T, F_L = 19, 15


def table(nid, col, row, name, lines):
    return c.box(nid, XS[col], ROWS[row], BW, 30 + 21 * len(lines), name,
                 lines, shape="table", f_title=F_T, f_line=F_L)


user = table("user", 0, 0, "app_user", [
    "user_id TEXT PK",
    "email TEXT UQ",
    "full_name TEXT NOT NULL",
    "role TEXT CHECK (resident|consultant|admin)",
    "is_active BOOLEAN DEFAULT true",
    "created_at TIMESTAMPTZ NOT NULL",
    "password_hash TEXT NOT NULL",
])

case = table("case", 1, 0, "case_file", [
    "case_id TEXT PK",
    "patient_ref TEXT NOT NULL",
    "organ TEXT NOT NULL",
    "status TEXT DEFAULT 'open'",
    "opened_by TEXT FK app_user",
    "opened_at TIMESTAMPTZ NOT NULL",
])

series = table("series", 2, 0, "scan_series", [
    "series_id TEXT PK",
    "case_id TEXT FK case_file",
    "scan_id TEXT UQ",
    "modality TEXT NOT NULL",
    "slice_count INTEGER NOT NULL",
    "stored_at TIMESTAMPTZ NOT NULL",
])

model = table("model", 0, 1, "model_registry", [
    "model_id TEXT PK",
    "organ TEXT NOT NULL",
    "task TEXT NOT NULL",
    "architecture TEXT NOT NULL",
    "version TEXT NOT NULL",
    "checkpoint_path TEXT NOT NULL",
])

report = table("report", 1, 1, "report_version", [
    "report_id TEXT PK",
    "case_id TEXT FK case_file",
    "version INTEGER NOT NULL",
    "author_id TEXT FK app_user",
    "signed_at TIMESTAMPTZ NULL",
    "pdf_path TEXT NULL",
])

result = table("result", 2, 1, "classification_result", [
    "result_id TEXT PK",
    "series_id TEXT FK scan_series",
    "model_id TEXT FK model_registry",
    "predicted_class TEXT NOT NULL",
    "confidence REAL NOT NULL",
    "label_space TEXT NOT NULL",
])

audit = table("audit", 0, 2, "audit_log", [
    "log_id BIGSERIAL PK",
    "actor_id TEXT FK app_user",
    "action TEXT NOT NULL",
    "entity TEXT NOT NULL",
    "entity_id TEXT NOT NULL",
    "occurred_at TIMESTAMPTZ NOT NULL",
])

mask = table("mask", 2, 2, "segmentation_mask", [
    "mask_id TEXT PK",
    "result_id TEXT FK classification_result",
    "model_id TEXT FK model_registry",
    "prompt_mode TEXT CHECK (auto|point|box)",
    "mask_path TEXT NOT NULL",
    "volume_mm3 REAL NULL",
])

# ---------------------------------------------------------------- relations ---
c.edge([(440, 232), (558, 232)], "opened_by", side="up")
c.edge([(940, 228), (1058, 228)], "case_id", side="up")
c.edge([(1250, 306), (1250, 428)], "series_id", side="l")
c.edge([(750, 306), (750, 428)], "case_id", side="l")
c.edge([(60, 238), (36, 238), (36, 768), (58, 768)], "actor_id", side="r")
c.edge([(250, 586), (250, 630), (1470, 630), (1470, 508), (1442, 508)],
       "model_id", side="up")
c.edge([(1250, 586), (1250, 688)], "result_id", side="l")

c.note(60, 900,
       "Indexes: UNIQUE(app_user.email) · UNIQUE(scan_series.scan_id) · "
       "UNIQUE(report_version.case_id, version) · INDEX(audit_log.occurred_at) "
       "· INDEX(classification_result.series_id)", size=16)
c.note(60, 926,
       "segmentation_mask.model_id carries the same foreign key to "
       "model_registry as classification_result does; one arrow is drawn to "
       "keep the figure readable.", size=16)

build(c, "database_openmed.svg")
