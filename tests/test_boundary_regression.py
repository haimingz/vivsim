"""Golden-value regression tests for all boundary condition functions.

These tests capture the exact output of each boundary function for a known
input, so that any refactoring that accidentally changes the physics will
be caught immediately.  Checksums were captured from the verified
implementation on 2026-04-04.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm

NX, NY = 8, 8


def _make_input():
    """Deterministic non-trivial distribution function."""
    key = jax.random.PRNGKey(42)
    rho = 1.0 + jax.random.uniform(key, (NX, NY), minval=-0.01, maxval=0.01)
    u = jax.random.uniform(jax.random.PRNGKey(7), (2, NX, NY), minval=-0.05, maxval=0.05)
    return lbm.get_equilibrium(rho, u)


# ── Golden checksums (sum of full output array) ─────────────────────────

_NEBB_CHECKSUMS = {
    "left": 64.0586013794, "right": 63.9197845459,
    "top": 64.1758270264, "bottom": 64.1786651611,
}
_NEE_CHECKSUMS = {
    "left": 64.0358123779, "right": 64.0435104370,
    "top": 64.0293502808, "bottom": 64.0384674072,
}
_EQ_CHECKSUMS = {
    "left": 64.0358123779, "right": 64.0435104370,
    "top": 64.0293502808, "bottom": 64.0384674072,
}
_BB_CHECKSUMS = {
    "left": 64.1100540161, "right": 63.9708442688,
    "top": 64.0411605835, "bottom": 64.0397186279,
}
_SR_CHECKSUMS = {
    "left": 64.1100540161, "right": 63.9708442688,
    "top": 64.0411605835, "bottom": 64.0397186279,
}
_VEL_NEBB_CHECKSUMS = {
    "left": 64.0588378906, "right": 63.9210166931,
    "top": 64.2158279419, "bottom": 64.1386642456,
}
_PRES_NEBB_CHECKSUMS = {
    "left": 63.9134025574, "right": 64.0514755249,
    "top": 64.0373458862, "bottom": 64.2308883667,
}

ATOL = 1e-3  # tolerance for float32 checksum comparison


class NEBBGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        for loc, expected in _NEBB_CHECKSUMS.items():
            f_out = lbm.boundary_nebb(f, loc=loc, ux_wall=0.01, uy_wall=0.005)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"NEBB checksum changed for {loc}")


class NEEGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        for loc, expected in _NEE_CHECKSUMS.items():
            f_out = lbm.boundary_nee(f, loc=loc, ux_wall=0.01)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"NEE checksum changed for {loc}")


class EquilibriumBCGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        for loc, expected in _EQ_CHECKSUMS.items():
            f_out = lbm.boundary_equilibrium(f, loc=loc, ux_wall=0.01)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"EQ checksum changed for {loc}")


class BounceBackGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        fs = lbm.streaming(f)
        for loc, expected in _BB_CHECKSUMS.items():
            f_out = lbm.boundary_bounce_back(f, fs, loc=loc,
                                             ux_wall=0.01, uy_wall=0.005)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"BB checksum changed for {loc}")


class SpecularReflectionGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        fs = lbm.streaming(f)
        for loc, expected in _SR_CHECKSUMS.items():
            f_out = lbm.boundary_specular_reflection(f, fs, loc=loc,
                                                     ux_wall=0.01, uy_wall=0.005)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"SR checksum changed for {loc}")


class VelocityNEBBGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        for loc, expected in _VEL_NEBB_CHECKSUMS.items():
            f_out = lbm.boundary_velocity_nebb(f, loc=loc, ux_wall=0.01)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"Vel NEBB checksum changed for {loc}")


class PressureNEBBGoldenTest(absltest.TestCase):
    def test_golden_checksums(self):
        f = _make_input()
        for loc, expected in _PRES_NEBB_CHECKSUMS.items():
            f_out = lbm.boundary_pressure_nebb(f, loc=loc, rho_wall=1.001)
            self.assertAlmostEqual(float(f_out.sum()), expected, delta=ATOL,
                                   msg=f"Pres NEBB checksum changed for {loc}")


if __name__ == "__main__":
    absltest.main()
