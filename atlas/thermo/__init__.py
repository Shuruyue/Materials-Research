# Make imports lazy or optional to avoid breaking if dependencies (like pycalphad) are missing
# when we just want to use OpenMM
try:
    from atlas.thermo.calphad import CalphadCalculator
except ImportError:
    CalphadCalculator = None

try:
    from atlas.thermo.stability import PhaseStabilityAnalyst, ReferenceDatabase
except ImportError:
    PhaseStabilityAnalyst = None
    ReferenceDatabase = None

__all__ = [
    "CalphadCalculator",
    "PhaseStabilityAnalyst",
    "ReferenceDatabase",
]
