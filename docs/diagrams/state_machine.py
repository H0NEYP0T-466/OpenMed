#!/usr/bin/env python3
"""OpenMed SRS — State Machine Diagram (Figure 5.6.7).

The organ lifecycle. One organ's weights are either absent from RAM, being read
off disk, resident and serving, being dropped, or gone again. Everything here
maps onto real code:

  pipeline.load()                -> Warming
  pipeline.is_loaded             -> Active
  pipeline.unload()              -> Unloading
  release_runtime_memory()       -> Released (torch.cuda.empty_cache, gc)
  current_rss_mb()               -> the number /health reports as proof

Run: python3 state_machine.py
Out: state_openmed.svg
"""

from svgkit import Canvas, build, INK

W, H = 1500, 980

c = Canvas(
    W, H,
    "OpenMed — Organ Lifecycle State Machine",
    subtitle="Rule 4 in the SRS keeps one organ resident at a time; "
             "this is the machine that enforces it",
    footer="A transition that raises leaves the organ in Unavailable, never "
           "half-loaded: the pipeline drops its partial state before "
           "the exception reaches the router.",
)

# ---------------------------------------------------------------- states ---
U = c.box("U", 90, 200, 300, 92, "Unloaded",
          ["no weights in RAM", "is_loaded is False"], shape="round")
WX = c.box("W", 560, 200, 320, 92, "Warming",
           ["state_dict read from disk", "labels and manifest parsed"],
           shape="round")
A = c.box("A", 1030, 200, 340, 92, "Active",
          ["forward pass served", "one organ only"], shape="round")
R = c.box("R", 560, 470, 320, 92, "Released",
          ["refs dropped, cache emptied", "RSS back near baseline"],
          shape="round")
UD = c.box("UD", 1030, 470, 340, 92, "Unloading",
           ["del model, gc.collect()"], shape="round")
X = c.box("X", 560, 720, 320, 92, "Unavailable",
          ["adapter reports available=False"], shape="round")

# initial pseudostate
c.raw(f'<circle cx="240" cy="150" r="11" fill="{INK}"/>')
c.edge([(240, 161), (240, 198)])

c.edge([(390, 246), (558, 246)],
       "POST /load-models", side="up")
c.edge([(880, 246), (1028, 246)],
       "weights resident", side="up")
c.edge([(1200, 292), (1200, 468)],
       "unload-models, or a second organ requested", side="l")
c.edge([(1030, 516), (882, 516)],
       "del + gc", side="up")
c.edge([(560, 516), (460, 516), (460, 640), (240, 640), (240, 294)],
       "release_runtime_memory()", side="up")
c.edge([(1370, 246), (1410, 246), (1410, 766), (882, 766)],
       "OOM or adapter error", dashed=True, side="up")
c.edge([(560, 766), (52, 766), (52, 246), (88, 246)],
       "operator retries", dashed=True, side="r")

c.note(60, 880,
       "Rule 4: only one organ is resident at a time. Selecting a different organ "
       "unloads the current one before the next checkpoint is read.", size=16)
c.note(60, 906,
       "/health reports current_rss_mb(), so an unload can be shown to have actually "
       "returned memory rather than merely dropped a reference.", size=16)

build(c, "state_openmed.svg")
