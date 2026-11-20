#!/usr/bin/env python3
"""svgkit — shared SVG figure engine for the OpenMed SRS diagrams.

One place for geometry, text metrics, the self-audit and the renderer, so every
figure in docs/diagrams/ is drawn and checked the same way.

Audit gate (raises before writing if it fails):
  * every label measures inside its node
  * no two nodes overlap
  * no arrow segment crosses a node it does not start or end at

Shapes: rect, round, ellipse, diamond, uml, table.
Arrow styles: v, h, elbow, fan, lmargin, diag.
Sequence support: lifeline() + msg().
"""

from PIL import ImageFont

__all__ = ["Canvas"]

FONT_REG = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"

INK = "#22323b"
TITLE_INK = "#15242c"
BODY_INK = "#37474f"
MUTED = "#4a5b63"
BAND_STROKE = "#c3d0d6"
ARROW_INK = "#31434c"


class Canvas:
    def __init__(self, w, h, title, subtitle=None, footer=None):
        self.w, self.h = w, h
        self.title = title
        self.subtitle = subtitle
        self.footer = footer
        self.bands = []
        self.nodes = []
        self.arrows = []
        self.edges = []
        self.raws = []
        self.lifelines = []
        self.msgs = []
        self.notes = []
        self._by_id = {}

    # ---------------------------------------------------------- content ---
    def band(self, y0, y1, title, tint="#eef3f5"):
        self.bands.append((y0, y1, title, tint))

    def box(self, nid, x, y, w, h, title, lines=(), shape="rect",
            f_title=20, f_line=16, attrs=None, methods=None, align="left"):
        node = dict(id=nid, x=x, y=y, w=w, h=h, title=title, lines=list(lines),
                    shape=shape, f_title=f_title, f_line=f_line,
                    attrs=list(attrs or []), methods=list(methods or []),
                    align=align)
        self.nodes.append(node)
        self._by_id[nid] = node
        return node

    def note(self, x, y, text, size=16, anchor="start", bold=False):
        self.notes.append((x, y, text, size, anchor, bold))

    def arrow(self, fr, to, label="", style="v", dashed=False):
        self.arrows.append(dict(fr=fr, to=to, label=label, style=style,
                                dashed=dashed))

    def edge(self, pts, label="", dashed=False, head="ar", tail=None,
             at=0.5, side=None):
        """Hand-routed polyline. `pts` are absolute waypoints.

        head/tail: None, "ar" (filled), "tri" (hollow triangle), "dia".
        `at` picks which fraction of the whole path carries the label.
        `side` forces the label above/below/left/right ("up"|"down"|"l"|"r").
        """
        self.edges.append(dict(pts=[(float(x), float(y)) for x, y in pts],
                               label=label, dashed=dashed, head=head,
                               tail=tail, at=at, side=side))

    def raw(self, fragment):
        """Escape hatch for the few glyphs the shape set does not cover."""
        self.raws.append(fragment)

    def lifeline(self, nid, x, y0, y1, title):
        self.lifelines.append(dict(id=nid, x=x, y0=y0, y1=y1, title=title))
        self._by_id[nid] = dict(id=nid, x=x, y=y0, w=1, h=y1 - y0,
                                title=title, lines=[], shape="lifeline",
                                f_title=20, f_line=16, attrs=[], methods=[],
                                align="center")

    def msg(self, fr, to, y, label, dashed=False, ret=False):
        self.msgs.append(dict(fr=fr, to=to, y=y, label=label, dashed=dashed,
                              ret=ret))

    # ---------------------------------------------------------- metrics ---
    @staticmethod
    def tw(text, size, bold=False):
        f = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
        return f.getlength(text)

    def _limits(self, n):
        """Usable text width inside a node, per shape."""
        shape = n["shape"]
        if shape == "ellipse":
            return n["w"] * 0.70
        if shape == "diamond":
            return n["w"] * 0.52
        return n["w"] - 24

    def _height_needed(self, n):
        shape = n["shape"]
        if shape == "uml":
            return max(96, 74 + (len(n["attrs"]) + len(n["methods"])) * 18)
        if shape == "table":
            return 30 + len(n["lines"]) * 21
        if shape in ("ellipse", "diamond"):
            return 30 + len(n["lines"]) * 22
        return 30 + len(n["lines"]) * 23

    # ------------------------------------------------------------ audit ---
    def audit(self, verbose=True):
        bad = []
        for n in self.nodes:
            if n["shape"] == "lifeline":
                continue
            lim = self._limits(n)
            if n["title"]:
                t = self.tw(n["title"], n["f_title"], True)
                if t > lim:
                    bad.append(f"{n['id']}: title overflows "
                               f"({t:.0f} > {lim:.0f}) '{n['title'][:26]}'")
            for ln in n["lines"] + n["attrs"] + n["methods"]:
                t = self.tw(ln, n["f_line"])
                if t > lim:
                    bad.append(f"{n['id']}: line overflows "
                               f"({t:.0f} > {lim:.0f}) '{ln[:26]}'")
            need = self._height_needed(n)
            if need > n["h"]:
                bad.append(f"{n['id']}: text needs {need}px, node is "
                           f"{n['h']}px")

        solid = [n for n in self.nodes if n["shape"] != "lifeline"]
        for i in range(len(solid)):
            for j in range(i + 1, len(solid)):
                a, b = solid[i], solid[j]
                if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                        and a["y"] < b["y"] + b["h"]
                        and b["y"] < a["y"] + a["h"]):
                    bad.append(f"{a['id']} overlaps {b['id']}")

        for a in self.arrows:
            pts = self.geom(a["fr"], a["to"], a["style"])
            for i in range(len(pts) - 1):
                for n in solid:
                    if n["id"] in (a["fr"], a["to"]):
                        continue
                    if _seg_hits(pts[i], pts[i + 1], n):
                        bad.append(f"arrow {a['fr']}->{a['to']} crosses "
                                   f"{n['id']}")

        for a in self.edges:
            pts = a["pts"]
            ends = set()
            for p in (pts[0], pts[-1]):
                for n in solid:
                    if (n["x"] - 2 <= p[0] <= n["x"] + n["w"] + 2
                            and n["y"] - 2 <= p[1] <= n["y"] + n["h"] + 2):
                        ends.add(n["id"])
            for i in range(len(pts) - 1):
                for n in solid:
                    if n["id"] in ends:
                        continue
                    if _seg_hits(pts[i], pts[i + 1], n):
                        bad.append(f"edge {pts[0]}->{pts[-1]} crosses "
                                   f"{n['id']}")

        for m in self.msgs:
            xs = sorted([self._by_id[m["fr"]]["x"],
                         self._by_id[m["to"]]["x"]])
            if xs[1] - xs[0] < 90:
                bad.append(f"msg {m['fr']}->{m['to']} too short")

        if verbose:
            for b in bad:
                print("  " + b)
        return bad

    # --------------------------------------------------------- geometry ---
    def geom(self, fr, to, style):
        a, b = self._by_id[fr], self._by_id[to]
        ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
        bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
        acx, bcx = ax + aw / 2, bx + bw / 2
        acy, bcy = ay + ah / 2, by + bh / 2

        if style == "fan":
            return [(bcx, ay + ah), (bcx, by)]
        if style == "h":
            if ax < bx:
                return [(ax + aw, acy), (bx, bcy)]
            return [(ax, acy), (bx + bw, bcy)]
        if style == "lmargin":
            mx = 40
            return [(ax, acy), (mx, acy), (mx, bcy), (bx, bcy)]
        if style == "rmargin":
            mx = self.w - 40
            return [(ax + aw, acy), (mx, acy), (mx, bcy), (bx + bw, bcy)]
        if style == "diag":
            return [(acx, ay + ah), (bcx, by)]
        if style == "ret":
            lane = by - 20
            return [(acx, ay), (acx, lane), (bcx, lane), (bcx, by)]
        if style in ("v", "elbow"):
            if abs(acx - bcx) < 6:
                if ay < by:
                    return [(acx, ay + ah), (bcx, by)]
                return [(acx, ay), (bcx, by + bh)]
            if ay < by:
                mid = (ay + ah + by) / 2
                return [(acx, ay + ah), (acx, mid), (bcx, mid), (bcx, by)]
            mid = (by + bh + ay) / 2
            return [(acx, ay), (acx, mid), (bcx, mid), (bcx, by + bh)]
        raise ValueError(style)

    # ----------------------------------------------------------- render ---
    def _esc(self, s):
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    def render(self, path):
        e = self._esc
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
             f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">',
             '<defs>'
             '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_INK}"/></marker>'
             '<marker id="aro" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="none" '
             'stroke="#31434c" stroke-width="1.4"/></marker>'
             '<marker id="tri" viewBox="0 0 12 12" refX="11" refY="6" '
             'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 12 6 L 0 12 z" fill="#ffffff" '
             'stroke="#31434c" stroke-width="1.4"/></marker>'
             '<marker id="dia" viewBox="0 0 14 12" refX="13" refY="6" '
             'markerWidth="11" markerHeight="10" orient="auto-start-reverse">'
             f'<path d="M 0 6 L 7 0 L 14 6 L 7 12 z" fill="{ARROW_INK}"/>'
             '</marker>'
             '</defs>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="60" y="62" font-family="DejaVu Sans, sans-serif" '
             f'font-size="32" font-weight="700" fill="#101820">'
             f'{e(self.title)}</text>']

        if self.subtitle:
            o.append(f'<text x="60" y="94" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="18" fill="{MUTED}">'
                     f'{e(self.subtitle)}</text>')

        for y0, y1, title, tint in self.bands:
            o.append(f'<rect x="40" y="{y0}" width="{self.w - 80}" '
                     f'height="{y1 - y0}" fill="{tint}" stroke="{BAND_STROKE}" '
                     'stroke-width="1.2"/>')
            o.append(f'<text x="56" y="{y0 - 8}" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="18" font-weight="700" '
                     f'fill="{MUTED}">{e(title)}</text>')

        # raw fragments sit under everything except bands and titles
        for r in self.raws:
            o.append(r)

        # lifelines first (behind everything)
        for ll in self.lifelines:
            o.append(f'<line x1="{ll["x"]}" y1="{ll["y0"]}" x2="{ll["x"]}" '
                     f'y2="{ll["y1"]}" stroke="#9aa9b2" stroke-width="1.4" '
                     'stroke-dasharray="6 5"/>')

        for a in self.arrows:
            pts = self.geom(a["fr"], a["to"], a["style"])
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            dash = ' stroke-dasharray="7 5"' if a["dashed"] else ""
            o.append(f'<path d="{d}" fill="none" stroke="{ARROW_INK}" '
                     f'stroke-width="1.7"{dash} marker-end="url(#ar)"/>')
            if a["label"]:
                p1, p2 = pts[0], pts[-1]
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                wd = self.tw(a["label"], 15)
                anchor = "start" if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]) \
                    else "middle"
                rx = mx - 3 if anchor == "start" else mx - wd / 2 - 3
                my += 5 if anchor == "start" else -9
                o.append(f'<rect x="{rx:.1f}" y="{my - 13:.1f}" '
                         f'width="{wd + 6:.1f}" height="18" fill="#ffffff" '
                         'opacity="0.93"/>')
                o.append(f'<text x="{mx:.1f}" y="{my:.1f}" '
                         f'text-anchor="{anchor}" font-family="DejaVu Sans, '
                         f'sans-serif" font-size="15" fill="#3d4f58">'
                         f'{e(a["label"])}</text>')

        # hand-routed edges
        for a in self.edges:
            pts = a["pts"]
            d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            dash = ' stroke-dasharray="7 5"' if a["dashed"] else ""
            mk = ""
            if a["tail"]:
                mk += f' marker-start="url(#{a["tail"]})"'
            if a["head"]:
                mk += f' marker-end="url(#{a["head"]})"'
            o.append(f'<path d="{d}" fill="none" stroke="{ARROW_INK}" '
                     f'stroke-width="1.7"{dash}{mk}/>')
            if not a["label"]:
                continue
            # label sits on the longest segment of the requested orientation
            side_pref = a["side"]
            want_h = side_pref in ("up", "down")
            best, blen = 0, -1
            for i in range(len(pts) - 1):
                dx = abs(pts[i + 1][0] - pts[i][0])
                dy = abs(pts[i + 1][1] - pts[i][1])
                L = (dx * dx + dy * dy) ** 0.5
                horiz = dx >= dy
                if horiz != want_h:
                    continue
                if L > blen:
                    best, blen = i, L
            if blen < 0:
                for i in range(len(pts) - 1):
                    L = ((pts[i + 1][0] - pts[i][0]) ** 2
                         + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
                    if L > blen:
                        best, blen = i, L
            i = best
            mx = (pts[i][0] + pts[i + 1][0]) / 2
            my = (pts[i][1] + pts[i + 1][1]) / 2
            horiz = abs(pts[i + 1][0] - pts[i][0]) >= abs(pts[i + 1][1] - pts[i][1])
            side = side_pref or ("up" if horiz else "r")
            wd = self.tw(a["label"], 15)
            bh = 18
            if side == "up":
                rx, ry = mx - wd / 2 - 3, my - 6 - bh
                tx, ty, anc = mx, my - 10, "middle"
            elif side == "down":
                rx, ry = mx - wd / 2 - 3, my + 6
                tx, ty, anc = mx, my + 19, "middle"
            elif side == "l":
                rx, ry = mx - wd - 9, my - bh / 2
                tx, ty, anc = mx - 6, my + 5, "end"
            else:
                rx, ry = mx + 6, my - bh / 2
                tx, ty, anc = mx + 9, my + 5, "start"
            o.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{wd + 6:.1f}" '
                     f'height="{bh}" fill="#ffffff" opacity="0.95"/>')
            o.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anc}" '
                     f'font-family="DejaVu Sans, sans-serif" font-size="15" '
                     f'fill="#3d4f58">{e(a["label"])}</text>')

        # sequence messages
        for m in self.msgs:
            x1 = self._by_id[m["fr"]]["x"]
            x2 = self._by_id[m["to"]]["x"]
            y = m["y"]
            dash = ' stroke-dasharray="7 5"' if m["dashed"] else ""
            o.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                     f'stroke="{ARROW_INK}" stroke-width="1.6"{dash} '
                     'marker-end="url(#ar)"/>')
            mid = (x1 + x2) / 2
            wd = self.tw(m["label"], 15)
            o.append(f'<rect x="{mid - wd / 2 - 4:.1f}" y="{y - 21}" '
                     f'width="{wd + 8:.1f}" height="18" fill="#ffffff" '
                     'opacity="0.95"/>')
            o.append(f'<text x="{mid:.1f}" y="{y - 7}" text-anchor="middle" '
                     f'font-family="DejaVu Sans, sans-serif" font-size="15" '
                     f'fill="#3d4f58">{e(m["label"])}</text>')

        for n in self.nodes:
            o.extend(self._node_svg(n, e))

        for x, y, text, size, anchor, bold in self.notes:
            weight = ' font-weight="700"' if bold else ""
            o.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
                     f'font-family="DejaVu Sans, sans-serif" font-size="{size}"'
                     f'{weight} fill="{MUTED}">{e(text)}</text>')

        if self.footer:
            o.append(f'<text x="60" y="{self.h - 22}" font-family="DejaVu '
                     f'Sans, sans-serif" font-size="16" fill="{MUTED}">'
                     f'{e(self.footer)}</text>')

        o.append('</svg>')
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(o))

    def _node_svg(self, n, e):
        o = []
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        shape = n["shape"]
        if shape == "lifeline":
            bw, bh = max(180, self.tw(n["title"], 18, True) + 30), 44
            bx = x - bw / 2
            o.append(f'<rect x="{bx:.1f}" y="{y - bh}" width="{bw:.1f}" '
                     f'height="{bh}" fill="#ffffff" stroke="{INK}" '
                     'stroke-width="1.8" rx="3"/>')
            o.append(f'<text x="{x}" y="{y - 15}" text-anchor="middle" '
                     f'font-family="DejaVu Sans, sans-serif" font-size="18" '
                     f'font-weight="700" fill="{TITLE_INK}">'
                     f'{e(n["title"])}</text>')
            return o

        if shape == "ellipse":
            o.append(f'<ellipse cx="{x + w / 2}" cy="{y + h / 2}" '
                     f'rx="{w / 2}" ry="{h / 2}" fill="#ffffff" '
                     f'stroke="{INK}" stroke-width="1.8"/>')
            o.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 6}" '
                     f'text-anchor="middle" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="{n["f_title"]}" '
                     f'font-weight="700" fill="{TITLE_INK}">'
                     f'{e(n["title"])}</text>')
            return o

        if shape == "diamond":
            cx, cy = x + w / 2, y + h / 2
            pts = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
            o.append(f'<polygon points="{pts}" fill="#ffffff" stroke="{INK}" '
                     'stroke-width="1.8"/>')
            yy = cy - 4 if not n["lines"] else cy - 14
            o.append(f'<text x="{cx}" y="{yy}" text-anchor="middle" '
                     f'font-family="DejaVu Sans, sans-serif" '
                     f'font-size="{n["f_title"]}" font-weight="700" '
                     f'fill="{TITLE_INK}">{e(n["title"])}</text>')
            for i, ln in enumerate(n["lines"]):
                o.append(f'<text x="{cx}" y="{yy + 20 + i * 20}" '
                         f'text-anchor="middle" font-family="DejaVu Sans, '
                         f'sans-serif" font-size="{n["f_line"]}" '
                         f'fill="{BODY_INK}">{e(ln)}</text>')
            return o

        if shape == "store":
            # Yourdon open-ended data store: left end closed, right end open
            o.append(f'<path d="M {x + w} {y} L {x} {y} L {x} {y + h} '
                     f'L {x + w} {y + h}" fill="none" stroke="{INK}" '
                     'stroke-width="1.8"/>')
            o.append(f'<rect x="{x}" y="{y}" width="44" height="{h}" '
                     'fill="#eef3f5" stroke="none"/>')
            o.append(f'<path d="M {x} {y} L {x} {y + h}" stroke="{INK}" '
                     'stroke-width="1.8"/>')
            o.append(f'<line x1="{x + 44}" y1="{y}" x2="{x + 44}" '
                     f'y2="{y + h}" stroke="{INK}" stroke-width="1.2"/>')
            o.append(f'<text x="{x + 22}" y="{y + h / 2 + 6}" '
                     'text-anchor="middle" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="17" font-weight="700" '
                     f'fill="{TITLE_INK}">{e(n["title"])}</text>')
            yy = y + 26
            for ln in n["lines"]:
                o.append(f'<text x="{x + 56}" y="{yy}" font-family="DejaVu '
                         f'Sans, sans-serif" font-size="{n["f_line"]}" '
                         f'fill="{BODY_INK}">{e(ln)}</text>')
                yy += 23
            return o

        # rect family
        rx = ' rx="4"' if shape in ("rect", "round", "uml", "table") else ""
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'fill="#ffffff" stroke="{INK}" stroke-width="1.8"{rx}/>')

        if shape == "uml":
            o.append(f'<text x="{x + w / 2}" y="{y + 26}" text-anchor="middle" '
                     f'font-family="DejaVu Sans, sans-serif" '
                     f'font-size="{n["f_title"]}" font-weight="700" '
                     f'fill="{TITLE_INK}">{e(n["title"])}</text>')
            o.append(f'<line x1="{x}" y1="{y + 38}" x2="{x + w}" '
                     f'y2="{y + 38}" stroke="{INK}" stroke-width="1.2"/>')
            yy = y + 56
            for a in n["attrs"]:
                o.append(f'<text x="{x + 10}" y="{yy}" font-family="DejaVu '
                         f'Sans, sans-serif" font-size="{n["f_line"]}" '
                         f'fill="{BODY_INK}">{e(a)}</text>')
                yy += 18
            yy += 4
            o.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" '
                     f'y2="{yy}" stroke="{INK}" stroke-width="1.2"/>')
            yy += 19
            for m in n["methods"]:
                o.append(f'<text x="{x + 10}" y="{yy}" font-family='
                         f'"DejaVu Sans, sans-serif" '
                         f'font-size="{n["f_line"]}" fill="{BODY_INK}">'
                         f'{e(m)}</text>')
                yy += 18
            return o

        if shape == "table":
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" '
                     f'fill="#eef3f5" stroke="{INK}" stroke-width="1.2"/>')
            o.append(f'<text x="{x + 10}" y="{y + 21}" font-family="DejaVu '
                     f'Sans, sans-serif" font-size="{n["f_title"]}" '
                     f'font-weight="700" fill="{TITLE_INK}">'
                     f'{e(n["title"])}</text>')
            yy = y + 30
            for i, ln in enumerate(n["lines"]):
                key = i % 2 == 0
                if key:
                    o.append(f'<rect x="{x}" y="{yy}" width="{w}" '
                             'height="21" fill="#fafafa" stroke="none"/>')
                o.append(f'<text x="{x + 10}" y="{yy + 15}" font-family='
                         f'"DejaVu Sans, sans-serif" '
                         f'font-size="{n["f_line"]}" fill="{BODY_INK}">'
                         f'{e(ln)}</text>')
                yy += 21
            return o

        anchor = "start"
        tx = x + 12
        if n["align"] == "center":
            anchor, tx = "middle", x + w / 2
        o.append(f'<text x="{tx}" y="{y + 27}" text-anchor="{anchor}" '
                 f'font-family="DejaVu Sans, sans-serif" '
                 f'font-size="{n["f_title"]}" font-weight="700" '
                 f'fill="{TITLE_INK}">{e(n["title"])}</text>')
        yy = y + 27 + 23
        for ln in n["lines"]:
            o.append(f'<text x="{tx}" y="{yy}" text-anchor="{anchor}" '
                     f'font-family="DejaVu Sans, sans-serif" '
                     f'font-size="{n["f_line"]}" fill="{BODY_INK}">'
                     f'{e(ln)}</text>')
            yy += 23
        return o


def _seg_hits(p1, p2, n):
    x0, y0 = n["x"] + 3, n["y"] + 3
    x1, y1 = n["x"] + n["w"] - 3, n["y"] + n["h"] - 3
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p1[0] - x0), (dx, x1 - p1[0]),
                 (-dy, p1[1] - y0), (dy, y1 - p1[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return t0 < t1 - 0.02 and t1 > 0.05 and t0 < 0.95


def build(canvas, path):
    bad = canvas.audit()
    print(f"defects={len(bad)}")
    if bad:
        raise SystemExit("audit failed — figure not written")
    canvas.render(path)
    print(f"wrote {path}")
