"""Minimal interactive keypoint editor for the packaged zoo SVGs.

Renders one animal's silhouette with its colored keypoints and skeleton, lets you drag points with
the mouse, and writes them back into the SVG. Deliberately trivial — no error handling, no undo.

Usage::

    uv run python examples/edit_zoo_keypoints.py duck

Called without an animal (or with an unknown one), it prints the list of packaged animals and exits.

Controls: press and hold the left mouse button on a dot to drag it, release to drop · ``s`` saves
back into the SVG (skeleton lines are regenerated from the new positions) · ``q`` (or closing the
window) quits. Saving always overwrites the animal's own SVG in place — no dialog, no prompt. The
color/name legend sits under the canvas.

"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Importing the package pulls in whichever augmentation backends are installed; albumentations then
# calls PyPI at import time, which warns for seconds on a machine with no CA bundle. Nothing here
# needs that check, so switch it off before the import rather than live with the noise.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# the schema lives in the package, not here — one definition for the loader, the writers and this editor
from fuse_augmentations.data.config import _KEYPOINT_COLORS, KEYPOINT_NAMES, KEYPOINT_SKELETON

ZOO_DIR = Path(__file__).resolve().parents[1] / "src" / "fuse_augmentations" / "data" / "zoo"
SVG_NS = "http://www.w3.org/2000/svg"
ZOO_NS = "https://github.com/Borda/fuse-augmentations/ns/zoo"
ET.register_namespace("", SVG_NS)
ET.register_namespace("zoo", ZOO_NS)

available = sorted(path.stem for path in ZOO_DIR.glob("*.svg"))
if len(sys.argv) < 2 or sys.argv[1] not in available:
    given = sys.argv[1] if len(sys.argv) > 1 else "(nothing)"
    print(f"unknown animal {given!r} — pick one of: {', '.join(available)}")
    raise SystemExit(1)
animal = sys.argv[1]
svg_path = ZOO_DIR / f"{animal}.svg"
root = ET.fromstring(svg_path.read_text())

outline_d = root.find(f"{{{SVG_NS}}}path[@id='outline']").get("d")
nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", outline_d)]
outline = list(zip(nums[0::2], nums[1::2], strict=False))

points: dict[str, list[float]] = {}
for circle in root.find(f"{{{SVG_NS}}}g[@id='keypoints']").findall(f"{{{SVG_NS}}}circle"):
    points[circle.get(f"{{{ZOO_NS}}}name")] = [float(circle.get("cx")), float(circle.get("cy"))]

# drop matplotlib's own "s = save figure" / "q = quit" bindings so this script owns those keys
plt.rcParams["keymap.save"] = []
plt.rcParams["keymap.quit"] = []

fig, ax = plt.subplots(figsize=(9, 9))
fig.canvas.manager.set_window_title(f"zoo keypoints — {animal}")
ax.add_patch(MplPolygon(outline, closed=True, facecolor="black", zorder=1))
ax.set_xlim(0, 1000)
ax.set_ylim(1000, 0)  # SVG y grows downward
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    f"{animal} — press & hold a dot to drag, release to drop  ·  s = save into SVG  ·  q = quit",
    fontsize=10,
)

skeleton_lines = []
for a, b in KEYPOINT_SKELETON:
    na, nb = KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]
    if na in points and nb in points:
        (line,) = ax.plot(
            [points[na][0], points[nb][0]],
            [points[na][1], points[nb][1]],
            color="#34c759",
            linewidth=2,
            zorder=2,
        )
        skeleton_lines.append((line, na, nb))

dots = {}
for name, (x, y) in points.items():
    (dot,) = ax.plot(
        x,
        y,
        "o",
        markersize=11,
        markerfacecolor=_KEYPOINT_COLORS[name],
        markeredgecolor="white",
        zorder=3,
        label=name,
    )
    dots[name] = dot

# color/name legend under the canvas instead of a label glued to every dot
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=6,
    fontsize=8,
    framealpha=0.9,
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)

dragging: list[str] = []


def refresh() -> None:
    for line, na, nb in skeleton_lines:
        line.set_data([points[na][0], points[nb][0]], [points[na][1], points[nb][1]])
    for name, dot in dots.items():
        dot.set_data([points[name][0]], [points[name][1]])
    fig.canvas.draw_idle()


def on_press(event) -> None:
    dragging.clear()
    if event.xdata is None:
        return
    best, best_d2 = None, 20.0**2
    for name, (x, y) in points.items():
        d2 = (x - event.xdata) ** 2 + (y - event.ydata) ** 2
        if d2 < best_d2:
            best, best_d2 = name, d2
    if best:
        dragging.append(best)


def on_drag(event) -> None:
    # true drag-and-drop: the dot follows only while the mouse button is held down
    if dragging and event.button == 1 and event.xdata is not None:
        points[dragging[0]][:] = [event.xdata, event.ydata]
        refresh()


def on_release(_event) -> None:
    dragging.clear()


def on_key(event) -> None:
    if event.key == "q":
        plt.close(fig)
    if event.key != "s":
        return
    for group_id in ("keypoints", "skeleton"):
        group = root.find(f"{{{SVG_NS}}}g[@id='{group_id}']")
        if group is not None:
            root.remove(group)
    skeleton = ET.Element(f"{{{SVG_NS}}}g", {"id": "skeleton"})
    for a, b in KEYPOINT_SKELETON:
        na, nb = KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]
        if na in points and nb in points:
            (x1, y1), (x2, y2) = points[na], points[nb]
            ET.SubElement(
                skeleton,
                f"{{{SVG_NS}}}line",
                {
                    "x1": f"{x1:.0f}",
                    "y1": f"{y1:.0f}",
                    "x2": f"{x2:.0f}",
                    "y2": f"{y2:.0f}",
                    "stroke": "#34c759",
                    "stroke-width": "3",
                    "stroke-linecap": "round",
                },
            )
    root.append(skeleton)
    group = ET.Element(f"{{{SVG_NS}}}g", {"id": "keypoints"})
    for name in KEYPOINT_NAMES:
        if name not in points:
            continue
        x, y = points[name]
        ET.SubElement(
            group,
            f"{{{SVG_NS}}}circle",
            {
                "id": f"kp-{name}",
                f"{{{ZOO_NS}}}name": name,
                "cx": f"{x:.0f}",
                "cy": f"{y:.0f}",
                "r": "8",
                "fill": _KEYPOINT_COLORS[name],
                "stroke": "#ffffff",
                "stroke-width": "1.5",
            },
        )
    root.append(group)
    out = ET.tostring(root, encoding="unicode")
    out = re.sub(r"><", ">\n  <", out)
    svg_path.write_text(out if out.endswith("\n") else out + "\n")
    fig.canvas.manager.set_window_title(f"zoo keypoints — {animal}  [saved]")


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_drag)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", on_key)
print("controls: press & hold a dot to drag, release to drop · s = save into SVG · q = quit")
plt.show()
