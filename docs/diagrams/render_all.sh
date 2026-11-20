#!/usr/bin/env bash
# Render every SRS figure in docs/diagrams/ to PNG (2600 px wide, for the
# compiled PDF/Word path) and to a vector PDF (for Overleaf).
#
#   ./render_all.sh              # rebuild every .svg that already exists
#   ./render_all.sh state        # rebuild one figure from its .py first
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -gt 0 ]]; then
    for stem in "$@"; do
        python3 "${stem}"_diagram.py 2>/dev/null || python3 "${stem}".py
    done
fi

shopt -s nullglob
for svg in *.svg; do
    name="${svg%.svg}"
    rsvg-convert -w 2600 -f png -o "${name}.png" "$svg"
    rsvg-convert -f pdf -o "${name}.pdf" "$svg"
    printf '%-34s png %7s  pdf %7s\n' "$svg" \
        "$(du -h "${name}.png" | cut -f1)" "$(du -h "${name}.pdf" | cut -f1)"
done
