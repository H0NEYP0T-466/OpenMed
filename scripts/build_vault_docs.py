#!/usr/bin/env python3
"""Build and refresh the OpenMed code documentation map in the Obsidian vault.

One page per source file, at the same path with ``.md`` appended:

    src/components/medical/OrganViewer3D.tsx
      -> OpenMed/src/components/medical/OrganViewer3D.tsx.md

Two kinds of page, and the distinction is enforced:

* **hand-written** (``generated: false``) — architecture notes, invariants,
  incident logs. This script NEVER overwrites one. If a source file already has
  a hand-written page, it is left exactly as it is.
* **generated** (``generated: true``) — purpose from the file's own docstring,
  exports with signatures, imports, and reverse dependencies. Refreshed
  whenever the source is newer than the page.

Usage
-----
    python scripts/build_vault_docs.py              # fill gaps + refresh stale
    python scripts/build_vault_docs.py --check      # report only, write nothing
    python scripts/build_vault_docs.py --force      # regenerate every generated page
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VAULT = Path("/home/honeypot/Obsidian/~Honeypot/OpenMed")

SKIP_DIRS = {
    "node_modules", "dist", "__pycache__", ".git", "venv", "datasets",
    "graphify-out", "For_Agents", ".ruff_cache", ".pytest_cache",
}
CODE_SUFFIXES = {".ts", ".tsx", ".css", ".py", ".mjs"}
ROOT_FILES = ("index.html", "package.json", "vite.config.ts",
              "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json")
# Byte-identical copies of the serving modules. Their docstrings come from the
# backend, so a page here is fine — but the *files* must never be restyled.
SNAPSHOTS = {"medsam_model.py", "preprocessor.py"}


# ─────────────────────────────────────────────────────────────────────────────
# Inventory and path mapping
# ─────────────────────────────────────────────────────────────────────────────

def collect_sources() -> list[Path]:
    found: list[Path] = []
    for pattern in ("src/**/*", "backend/app/**/*", "scripts/*"):
        for path in REPO.glob(pattern):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix in CODE_SUFFIXES:
                found.append(path)
    for name in ROOT_FILES:
        candidate = REPO / name
        if candidate.is_file():
            found.append(candidate)
    return sorted(set(found))


def doc_path(source: Path) -> Path:
    """Repo path -> vault path, with .md appended to the full filename."""
    rel = source.relative_to(REPO)
    parts = list(rel.parts)
    if len(parts) == 1:                       # repo-root file
        return Path("root") / (parts[0] + ".md")
    return Path(*parts[:-1]) / (parts[-1] + ".md")


def vault_link(target: Path) -> str:
    """Vault-absolute wikilink target, no .md extension."""
    return "OpenMed/" + str(target.with_suffix("")).replace(os.sep, "/")


def frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────────────────────────────────────
# Import resolution, for reverse dependencies
# ─────────────────────────────────────────────────────────────────────────────

def ts_specifiers(text: str) -> list[str]:
    out = re.findall(r"""from\s+['"](\.[^'"]+)['"]""", text)
    out += re.findall(r"""import\s+['"](\.[^'"]+)['"]""", text)
    return out


def py_specifiers(path: Path, text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                out.append(str(base / (node.module or "").replace(".", "/")))
            elif node.module and node.module.startswith("app."):
                out.append(str(REPO / "backend" / node.module.replace(".", "/")))
    return out


def resolve_ts(spec: str, src: Path) -> Path | None:
    base = (src.parent / spec).resolve()
    for candidate in (base.with_suffix(".tsx"), base.with_suffix(".ts"),
                      base / "index.ts", base / "index.tsx"):
        if candidate.is_file():
            return candidate
    return None


def resolve_py(base_str: str) -> Path | None:
    base = Path(base_str)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def leading_comment(text: str) -> str:
    match = re.match(r"\s*/\*\*(.*?)\*/", text, re.S)
    if match:
        lines = [re.sub(r"^\s*\*\s?", "", ln).rstrip() for ln in match.group(1).splitlines()]
        return "\n".join(lines).strip()
    match = re.match(r"\s*/\*(.*?)\*/", text, re.S)
    if match:
        lines = [re.sub(r"^\s*\*?\s?", "", ln).rstrip() for ln in match.group(1).splitlines()]
        return "\n".join(lines).strip()
    return ""


def ts_symbols(text: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for m in re.finditer(r"^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*(\([^)]*\))", text, re.M):
        found.append((m.group(1), "function", m.group(2)[:110]))
    for m in re.finditer(r"^export\s+interface\s+(\w+)", text, re.M):
        found.append((m.group(1), "interface", ""))
    for m in re.finditer(r"^export\s+type\s+(\w+)", text, re.M):
        found.append((m.group(1), "type", ""))
    for m in re.finditer(r"^export\s+const\s+(\w+)\s*(?::\s*([^=\n]{0,90}))?", text, re.M):
        found.append((m.group(1), "const", (m.group(2) or "").strip()[:90]))
    seen, unique = set(), []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique


def py_symbols(text: str) -> tuple[list[tuple[str, str, str]], str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], ""
    doc = ast.get_docstring(tree) or ""
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            symbols.append((node.name, kind, "(" + ", ".join(args[:6]) + ")"))
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases][:3]
            symbols.append((node.name, "class", "(" + ", ".join(bases) + ")" if bases else ""))
    return symbols, doc


def css_sections(text: str) -> list[str]:
    sections = [m.group(1).strip() for m in re.finditer(r"/\*\s*-{2,}\s*(.+?)\s*-{2,}\s*\*/", text)]
    if not sections:
        sections = [m.group(1).strip() for m in re.finditer(r"/\*\s*(.{4,80}?)\s*\*/", text)]
    return sections[:40]


def css_selectors(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"^([.#@a-zA-Z][^{}\n]{0,90}?)\s*\{", text, re.M):
        sel = m.group(1).strip()
        if sel.startswith("@") and "media" not in sel and "keyframes" not in sel:
            continue
        out.append(sel)
    return out[:60]


# ─────────────────────────────────────────────────────────────────────────────
# Page rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_page(source: Path, reverse: dict[Path, set[Path]]) -> str:
    text = source.read_text(errors="ignore")
    rel = source.relative_to(REPO)
    lines = text.count("\n") + 1

    if source.suffix == ".py":
        symbols, doc = py_symbols(text)
        doc = doc or leading_comment(text)
        kind = "python-module"
    elif source.suffix in {".ts", ".tsx"}:
        symbols, doc = ts_symbols(text), leading_comment(text)
        kind = "react-component" if source.suffix == ".tsx" else "typescript-module"
    elif source.suffix == ".css":
        symbols, doc = [], leading_comment(text)
        kind = "stylesheet"
    else:
        symbols, doc = [], ""
        kind = "config"

    out = ["---", f"source: {rel}", f"type: {kind}", f"lines: {lines}",
           "generated: true", "---", "", f"# `{rel}`", ""]

    crumbs = ["[[OpenMed/INDEX|OpenMed]]"]
    acc = Path("OpenMed")
    for part in doc_path(source).parts[:-1]:
        acc = acc / part
        crumbs.append(f"[[{acc}/INDEX|{part}]]")
    out += [" · ".join(crumbs), ""]

    if doc:
        out += ["## Purpose", "", doc[:1800], ""]

    if source.suffix == ".css":
        sections = css_sections(text)
        if sections:
            out += ["## Sections", ""] + [f"- {s}" for s in sections] + [""]
        selectors = css_selectors(text)
        if selectors:
            out += [f"## Top-level selectors ({len(selectors)})", "", "```"]
            out += selectors[:30]
            if len(selectors) > 30:
                out.append(f"… {len(selectors) - 30} more")
            out += ["```", ""]
    elif symbols:
        out += [f"## Exports ({len(symbols)})", "", "| Symbol | Kind | Signature |", "|---|---|---|"]
        for name, skind, sig in symbols:
            out.append(f"| `{name}` | {skind} | `{(sig or '').replace('|', chr(92) + '|')[:100]}` |")
        out.append("")

    specs = ts_specifiers(text) if source.suffix in {".ts", ".tsx"} else (
        py_specifiers(source, text) if source.suffix == ".py" else [])
    if specs:
        out += ["## Imports", ""] + [f"- `{s}`" for s in sorted(set(specs))[:24]] + [""]

    dependents = sorted(reverse.get(source, set()))
    if dependents:
        out += [f"## Used by ({len(dependents)})", ""]
        for dep in dependents[:20]:
            out.append(f"- [[{vault_link(doc_path(dep))}|{dep.relative_to(REPO)}]]")
        out.append("")

    out += ["---", "",
            "*Structural page generated from the source file — docstring, exports, imports and "
            "reverse dependencies. For narrative detail see [[OpenMed/architecture/INDEX|Architecture Notes]].*", ""]
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Folder indexes
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_indexes() -> int:
    pages = {p.relative_to(VAULT): p.read_text()
             for p in VAULT.rglob("*.md") if p.name != "INDEX.md"}
    if not pages:
        return 0

    dirs = set()
    for rel in pages:
        d = rel.parent
        while True:
            dirs.add(d)
            if d == Path("."):
                break
            d = d.parent

    written = 0
    for d in sorted(dirs):
        if d == Path("."):
            continue
        own = sorted([r for r in pages if r.parent == d], key=lambda r: r.name)
        subs = sorted({r.relative_to(d).parts[0] for r in pages
                       if r.parent != d and d in r.parents
                       and len(r.relative_to(d).parts) > 1})
        if not own and not subs:
            continue

        crumbs = ["[[OpenMed/INDEX|OpenMed]]"]
        acc = Path("OpenMed")
        for part in d.parts:
            acc = acc / part
            crumbs.append(f"[[{acc}/INDEX|{part}]]")

        out = ["---", f"folder: {d}", "generated: true", "---", "",
               f"# 📁 `{d}/`", "", " · ".join(crumbs), ""]
        if subs:
            out += ["## Subfolders", ""]
            out += [f"- [[OpenMed/{d}/{s}/INDEX|{s}/]]" for s in subs]
            out.append("")
        if own:
            out += [f"## Files ({len(own)})", "", "| Doc | Source | Type | Lines |", "|---|---|---|---|"]
            for r in own:
                t = pages[r]
                out.append(f"| [[{vault_link(r)}|{r.name[:-3]}]] | "
                           f"`{frontmatter_value(t, 'source') or '—'}` | "
                           f"{frontmatter_value(t, 'type') or '—'} | "
                           f"{frontmatter_value(t, 'lines') or ''} |")
            out.append("")
        out += ["---", "", "*Generated folder index. One page per source file.*", ""]
        (VAULT / d / "INDEX.md").write_text("\n".join(out))
        written += 1
    return written


def rebuild_master(n_hand: int, n_gen: int, n_src: int) -> None:
    top = sorted({r.parts[0] for r in
                  (p.relative_to(VAULT) for p in VAULT.rglob("*.md") if p.name != "INDEX.md")
                  if len(r.parts) > 1})
    labels = {
        "src": "`src/` — React frontend (TSX / TS / CSS)",
        "backend": "`backend/app/` — FastAPI backend (Python)",
        "scripts": "`scripts/` — dataset audits, cleaning, weight prep",
        "root": "repo root — `index.html`, `package.json`, Vite + TS config",
        "architecture": "**cross-cutting notes** — not tied to a single file",
        "superseded": "**historical** — describes code that no longer exists",
        "assets": "charts and generated HTML",
    }
    out = [
        "---",
        "title: OpenMed — Code Documentation Map",
        f"repo: {REPO}",
        "generated: true",
        "---",
        "",
        "# 🏥 OpenMed — Code Documentation Map",
        "",
        "> *One page per source file, mirroring the repository's folder structure.*",
        "",
        "| | |",
        "|---|---|",
        f"| Repository | `{REPO}` |",
        f"| Source files documented | **{n_src}** |",
        f"| Hand-written deep-dives | {n_hand} |",
        f"| Structural pages | {n_gen} |",
        "",
        "## How This Map Works",
        "",
        "**Every source file has exactly one page**, at the same path with `.md` appended:",
        "",
        "```",
        "src/components/medical/OrganViewer3D.tsx",
        "  → OpenMed/src/components/medical/OrganViewer3D.tsx.md",
        "```",
        "",
        "Two kinds of page:",
        "",
        "- **Hand-written** (`generated: false`) — architecture, invariants, bug incident logs. "
        "Written for the components that carry real complexity. The generator never overwrites these.",
        "- **Structural** (`generated: true`) — purpose from the file's own docstring, exports with "
        "signatures, imports, and **reverse dependencies** (who imports this file). Accurate, but "
        "not a deep-dive.",
        "",
        "## Top-Level Folders",
        "",
        "| Folder | Mirrors |",
        "|---|---|",
    ]
    out += [f"| [[OpenMed/{t}/INDEX|{t}/]] | {labels.get(t, '—')} |" for t in top]
    out += [
        "",
        "## Start Here",
        "",
        "| | |",
        "|---|---|",
        "| The 3D organ plate | [[OpenMed/src/components/medical/OrganViewer3D.tsx]] |",
        "| Dashboard that owns state | [[OpenMed/src/features/organs/OrganDashboard.tsx]] |",
        "| Brain classification workspace | [[OpenMed/src/features/organs/brain/BrainClassificationWorkspace.tsx]] |",
        "| Design system | [[OpenMed/architecture/design-system-atelier-zero]] |",
        "| CSS architecture | [[OpenMed/architecture/css-architecture]] |",
        "| Brain classifier invariants | [[OpenMed/backend/app/organs/brain/classification/INVARIANTS]] |",
        "| Segmentation serving contract | [[OpenMed/backend/app/organs/brain/segmentation/pipeline.py]] |",
        "| Kaggle training entry point | [[OpenMed/backend/app/organs/brain/segmentation/kaggle/trainKaggle.py]] |",
        "| Training package test suite | [[OpenMed/backend/app/organs/brain/segmentation/kaggle/test_medsam_pkg.py]] |",
        "",
        "## Project-Level Context",
        "",
        "This folder documents **code**. For the project itself — decisions, datasets, experiments, "
        "research — see the separate vault:",
        "",
        "- [[FYP/INDEX|FYP — project vault]]",
        "- [[Isabella_Brain/brain|Isabella's brain]]",
        "",
        "---",
        "",
        f"*Regenerate with `python scripts/build_vault_docs.py`. Last run: {n_src} source files.*",
        "",
    ]
    (VAULT / "INDEX.md").write_text("\n".join(out))


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(sources: list[Path]) -> tuple[int, int]:
    root = VAULT.parent
    by_path = {str(p.relative_to(root).with_suffix("")).replace(os.sep, "/"): p
               for p in root.rglob("*.md")}
    by_name: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*.md"):
        by_name[p.stem].append(p)

    link_re = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
    total = broken = 0
    examples = []
    for page in VAULT.rglob("*.md"):
        for raw in link_re.findall(page.read_text()):
            target = raw.strip()
            total += 1
            if target in by_path or target in by_name or f"OpenMed/{target}" in by_path \
                    or target.split("/")[-1] in by_name:
                continue
            broken += 1
            if len(examples) < 8:
                examples.append(f"{page.relative_to(VAULT)} -> [[{target}]]")

    missing = [s for s in sources if not (VAULT / doc_path(s)).is_file()]
    print(f"  source files    : {len(sources)}")
    print(f"  missing pages   : {len(missing)}")
    for m in missing:
        print("     MISSING:", m.relative_to(REPO))
    print(f"  wikilinks       : {total}")
    print(f"  broken links    : {broken}")
    for e in examples:
        print("     ", e)
    return len(missing), broken


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report only, write nothing")
    parser.add_argument("--force", action="store_true", help="regenerate every generated page")
    args = parser.parse_args()

    if not VAULT.is_dir():
        print(f"vault not found: {VAULT}", file=sys.stderr)
        return 2

    sources = collect_sources()
    print(f"source files: {len(sources)}")

    reverse: dict[Path, set[Path]] = defaultdict(set)
    for src in sources:
        text = src.read_text(errors="ignore")
        specs = ts_specifiers(text) if src.suffix in {".ts", ".tsx"} else (
            py_specifiers(src, text) if src.suffix == ".py" else [])
        for spec in specs:
            target = resolve_ts(spec, src) if src.suffix in {".ts", ".tsx"} else resolve_py(spec)
            if target and target in set(sources):
                reverse[target].add(src)

    created = refreshed = skipped = 0
    for src in sources:
        out = VAULT / doc_path(src)
        exists = out.is_file()

        if exists:
            existing = out.read_text(errors="ignore")
            block = existing.split("---", 2)[1] if existing.startswith("---") else ""
            if frontmatter_value(block, "generated") == "false":
                skipped += 1          # hand-written: never overwritten
                continue
            if not args.force and out.stat().st_mtime >= src.stat().st_mtime:
                skipped += 1
                continue
            action = "refresh"
        else:
            action = "create"

        if args.check:
            print(f"  would {action}: {doc_path(src)}")
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_page(src, reverse))
        if action == "create":
            created += 1
        else:
            refreshed += 1

    print(f"created {created}   refreshed {refreshed}   left alone {skipped}")

    if args.check:
        print("\n--check: nothing written")
        verify(sources)
        return 0

    n_indexes = rebuild_indexes()

    def _flag(page: Path) -> str:
        """Read the `generated:` value from the frontmatter block only.

        Grepping the whole file is wrong: the phrase appears in prose in the
        master index and in this script's own docstring.
        """
        text = page.read_text(errors="ignore")
        block = text.split("---", 2)[1] if text.startswith("---") else ""
        return frontmatter_value(block, "generated")

    pages = [p for p in VAULT.rglob("*.md") if p.name != "INDEX.md"]
    hand = sum(1 for p in pages if _flag(p) == "false")
    gen = sum(1 for p in pages if _flag(p) == "true")
    rebuild_master(hand, gen, len(sources))
    print(f"folder indexes rebuilt: {n_indexes}")
    print(f"master INDEX.md rebuilt")

    print("\nverification:")
    missing, broken = verify(sources)
    if missing or broken:
        print("\nFAILED — fix the above before trusting the map")
        return 1
    print("\nOK — full coverage, no broken links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
