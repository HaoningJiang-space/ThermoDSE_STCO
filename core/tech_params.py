"""Technology assumptions, factored out of values that used to be hardcoded
directly inside gen_hw_setting.py.

ThermoDSE's search tools (tools/scbo_search.py, rl_opt/*) explore the
*architecture* space (chiplet grid, systolic array size, buffer size, NoP/DRAM
bandwidth) under a single, fixed technology assumption: a specific D2D
interconnect energy and a specific defect-density/yield model, both previously
inlined as bare constants. This module gives those assumptions names and lets
a caller vary them explicitly, so a question like "does the optimal
architecture change if D2D interconnect energy improves?" can be asked without
editing simulation code -- only the TechParams passed in changes.

Every default below reproduces the exact value that was previously hardcoded,
so any caller that does not pass a TechParams sees identical behavior to
before this module existed.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class D2DParams:
    """Die-to-die (NoP) interconnect technology.

    Historical default (energy_per_bit_pJ=1.17) reproduces the value
    previously hardcoded in nop_setting_gen, chosen inside the [0.8, 1.3]
    pJ/bit range reported for Simba's GRS interconnect (Shao et al., MICRO
    2019, real fabricated 16nm silicon).
    """
    energy_per_bit_pJ: float = 1.17
    bandwidth_density_GBps_per_pin: float = 25.0
    pin_area_um2: float = 403 * 202


@dataclass(frozen=True)
class YieldParams:
    """Negative-binomial (Murphy/Seeds) defect-density yield model:

        Y = (1 + area_cm2 * defect_density_per_cm2 / cluster_alpha) ** (-cluster_alpha)

    Historical default (D0=0.08/cm^2, alpha=10) reproduces the 14nm-class
    value previously hardcoded in yield_setting_gen. Published references for
    other process/maturity points: ECO-CHIP (HPCA 2024) reports D0 in
    0.07-0.3 defects/cm^2 depending on process maturity; Chiplet Cloud
    reports D0=0.1/cm^2 at TSMC 7nm.
    """
    defect_density_per_cm2: float = 0.08
    cluster_alpha: float = 10.0


@dataclass(frozen=True)
class TechParams:
    """Full technology-assumption bundle threaded through the hardware cost
    model. `lib_type` is passed straight through to the existing
    mtxu/vecu/sram/regf_setting_gen functions in gen_hw_setting.py, which
    already accept it -- it is included here so all technology knobs live in
    one place, not because this module changes how it's used.
    """
    d2d: D2DParams = field(default_factory=D2DParams)
    yield_model: YieldParams = field(default_factory=YieldParams)
    lib_type: str = '28nm'
