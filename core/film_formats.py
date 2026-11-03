"""Film gauge table, from the reference engine's ``FotufilmCore/FilmFormat.swift``.

A gauge is a **short-edge millimetre length plus an aspect ratio** -- the reference models the
frame's short side (``frameHeightMM``) and derives everything else from it.  The sampling density
is anchored on the short edge too (``UnexposedEdge.pixelsPerMM`` divides the buffer's short edge
in pixels by ``frameHeightMM``), which is what makes a panoramic crop of the same gauge carry the
same grain per millimetre of film as a 3:2 one.

Our ``film_format_mm`` parameter stores the *long* edge -- 35 / 16 / 60 / 100 are the values old
``.sdc`` sidecars already carry, so the numbers below are keyed by them and the long edges are
corrected to the reference's real dimensions:

=================  ==============  ====================  ==========================
UI entry           key (long mm)   reference frame       was wrong by
=================  ==============  ====================  ==========================
35mm 标准全画幅    35.0            36 x 24, ratio 3:2    2.9 % (35 vs 36)
16mm 独立电影卷    16.0            12.5 x 7.4 Super 16   22 % (16 vs 12.5)
120 中画幅 (6x6)   60.0            56 x 56 image area    7 % (60 is the paper width)
4x5 大画幅         100.0           120 x 95 sheet        17 % (100 vs 120)
=================  ==============  ====================  ==========================
"""
from __future__ import annotations

# key: the stored ``film_format_mm`` value (long edge, historical).
# value: (short edge mm, aspect ratio long/short) -- FotufilmCore's FilmFormat table verbatim.
FORMAT_GAUGES: dict[float, tuple[float, float]] = {
    35.0: (24.0, 36.0 / 24.0),        # 135 still frame
    16.0: (7.4, 12.5 / 7.4),          # Super 16 motion picture frame
    60.0: (56.0, 1.0),                # 120 roll film, 6x6 image area
    100.0: (95.0, 120.0 / 95.0),      # 4x5 inch sheet film
}

DEFAULT_GAUGE = (24.0, 36.0 / 24.0)


def frame_short_mm(fmt_mm: float) -> float:
    """The gauge's short edge in millimetres for a stored ``film_format_mm`` value."""
    return FORMAT_GAUGES.get(float(fmt_mm), DEFAULT_GAUGE)[0]


def pixels_per_mm(width: int, height: int, fmt_mm: float) -> float:
    """Film pixels per millimetre, anchored on the frame's **short** edge.

    Matches ``UnexposedEdge.pixelsPerMM``: the buffer's short edge in pixels covers the gauge's
    short edge in millimetres.  Anchoring on the long edge instead made every non-3:2 picture --
    panoramas, squares -- carry a different grain scale than the gauge physically has.
    """
    short_px = max(min(int(width), int(height)), 1)
    return short_px / max(frame_short_mm(fmt_mm), 1e-6)
