"""Shared lookup tables for boundary slice and direction index dispatch.

All values are plain Python tuples/ints (not JAX arrays) so they are safe
to use as static values in JIT-traced contexts.
"""

VALID_LOCS = ("left", "right", "top", "bottom")

# Tuple indices for f.at[...] at the wall and the nearest interior neighbor.
# These include the leading direction axis (slice(None)) so they can be used
# directly as f[WALL_SLICE[loc]] to get all 9 directions at the wall.
WALL_SLICE = {
    "left":   (slice(None), 0),
    "right":  (slice(None), -1),
    "top":    (slice(None), slice(None), -1),
    "bottom": (slice(None), slice(None), 0),
}

NEIGHBOR_SLICE = {
    "left":   (slice(None), 1),
    "right":  (slice(None), -2),
    "top":    (slice(None), slice(None), -2),
    "bottom": (slice(None), slice(None), 1),
}

# Spatial-only wall slice (without the leading direction axis).
# Use as f[dir_idx][SPATIAL_WALL_SLICE[loc]] or f[(dir_idx,) + SPATIAL_WALL_SLICE[loc]].
SPATIAL_WALL_SLICE = {
    "left":   (0,),
    "right":  (-1,),
    "top":    (slice(None), -1),
    "bottom": (slice(None), 0),
}

SPATIAL_NEIGHBOR_SLICE = {
    "left":   (1,),
    "right":  (-2,),
    "top":    (slice(None), -2),
    "bottom": (slice(None), 1),
}

# Which axis of f gives the boundary length (for scalar-to-array expansion).
BOUNDARY_SIZE_AXIS = {
    "left": 2, "right": 2,    # boundary runs along NY
    "top": 1, "bottom": 1,    # boundary runs along NX
}

# ── NEBB / bounce-back / specular reflection configuration ───────────────
#
# Each wall is described by:
#   incoming   – 3 direction indices being set (unknown after streaming)
#   source     – 3 direction indices providing the known outgoing populations
#   tangential – 2 "known" tangential direction indices (for NEBB correction)
#   normal_sign – +1 for left/bottom (inward normal is +x / +y),
#                 -1 for right/top   (inward normal is -x / -y)
#   normal_axis – 0 (x-walls) or 1 (y-walls)
#
# The unified NEBB formula (verified against all 4 branches):
#   sn = normal_sign;  un = normal velocity;  ut = tangential velocity
#   f[i0, ws] = f[s0, ws]  +  sn * 2/3 * un * rho
#   f[i1, ws] = f[s1, ws]  -  sn * 0.5*(f[t0,ws] - f[t1,ws])
#                           +  sn * (1/6*un + 0.5*ut) * rho
#   f[i2, ws] = f[s2, ws]  +  sn * 0.5*(f[t0,ws] - f[t1,ws])
#                           +  sn * (1/6*un - 0.5*ut) * rho

NEBB_CONFIG = {
    "left": {
        "incoming": (1, 5, 8),
        "source": (3, 7, 6),
        "tangential": (2, 4),
        "normal_sign": 1,
        "normal_axis": 0,
    },
    "right": {
        "incoming": (3, 7, 6),
        "source": (1, 5, 8),
        "tangential": (2, 4),
        "normal_sign": -1,
        "normal_axis": 0,
    },
    "top": {
        "incoming": (4, 7, 8),
        "source": (2, 5, 6),
        "tangential": (1, 3),
        "normal_sign": -1,
        "normal_axis": 1,
    },
    "bottom": {
        "incoming": (2, 5, 6),
        "source": (4, 7, 8),
        "tangential": (1, 3),
        "normal_sign": 1,
        "normal_axis": 1,
    },
}

# ── Density / velocity recovery from known populations ───────────────────
#
# For velocity BCs: rho_wall = (sum of zero-momentum dirs + 2 * sum of
# outgoing dirs) / (1 + sign * un)
#
# For pressure BCs: un_wall = (sum of zero-momentum dirs + 2 * sum of
# outgoing dirs) / rho_wall - 1

DENSITY_RECOVERY = {
    "left":   {"zero": (0, 2, 4), "outgoing": (3, 6, 7), "sign": -1, "normal_axis": 0},
    "right":  {"zero": (0, 2, 4), "outgoing": (1, 5, 8), "sign":  1, "normal_axis": 0},
    "top":    {"zero": (0, 1, 3), "outgoing": (2, 5, 6), "sign":  1, "normal_axis": 1},
    "bottom": {"zero": (0, 1, 3), "outgoing": (4, 7, 8), "sign": -1, "normal_axis": 1},
}


def validate_loc(loc: str) -> None:
    """Raise ``ValueError`` if *loc* is not a valid boundary name."""
    if loc not in VALID_LOCS:
        raise ValueError(
            f"Boundary location `loc` should be one of {VALID_LOCS!r}, got {loc!r}."
        )
