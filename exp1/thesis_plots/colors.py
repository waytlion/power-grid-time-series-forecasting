"""Shared colorblind-safe categorical palette for every exp1 thesis figure.

8-hue order validated (CVD + normal-vision Delta E, adjacent-pairs) via the
dataviz skill's palette/validator -- reuse this module instead of matplotlib's
default cycle or ad-hoc hex so every figure in thesis_plots/ reads consistently.
"""

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
PALETTE_ORDER = list(PALETTE.values())

# Computer Modern to match the article-class thesis body font (cmr10 ships with matplotlib, no LaTeX needed)
THESIS_RC = {
    "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
}

OPT_LABELS = {"ac": "AC", "dc": "DC", "surrogate": "Surrogate", "e2e": "E2E"}
OPT_ORDER = ["ac", "dc", "surrogate", "e2e"]

# fixed identity -> hue, so the same arm reads the same color in every figure
ARM_COLORS = {
    "ac": PALETTE["blue"],
    "dc": PALETTE["orange"],
    "surrogate": PALETTE["aqua"],
    "e2e": PALETTE["yellow"],
}
_FALLBACK_SLOTS = [c for c in PALETTE_ORDER if c not in ARM_COLORS.values()]


def arm_color(label: str):
    """Resolve a color from a known optimizer keyword found in `label` (case-insensitive
    substring match, e.g. "AC-2step" -> ac's blue). Returns None if nothing matches."""
    key = label.lower()
    for opt, color in ARM_COLORS.items():
        if opt in key:
            return color
    return None


def build_color_map(labels) -> dict:
    """Color for each label: known optimizer keyword wins; unrecognized labels fall back to
    the palette's remaining slots, assigned in first-seen order (stable within one call)."""
    cmap, fallback_i = {}, 0
    for lb in labels:
        color = arm_color(lb)
        if color is None:
            color = _FALLBACK_SLOTS[fallback_i % len(_FALLBACK_SLOTS)]
            fallback_i += 1
        cmap[lb] = color
    return cmap
