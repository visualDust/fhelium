"""The degree-twelve power polynomial and its equivalent evaluation methods.

This workload measures polynomial evaluation, not an approximation to a named
activation function. Coefficients are in ascending power order on [-1, 1].
"""

from fhelium.experimental.bootstrap import (
    BalancedPowerEvaluator,
    HornerPowerEvaluator,
    PatersonStockmeyerPowerEvaluator,
    PolynomialApproximation,
)

COEFFICIENTS = (
    0.01, 0.2, -0.3, 0.125, -0.0625, 0.03125, -0.015625,
    0.0078125, -0.00390625, 0.001953125, -0.0009765625,
    0.00048828125, -0.000244140625,
)
POLYNOMIAL = PolynomialApproximation(
    basis="power", coefficients=COEFFICIENTS, domain=(-1.0, 1.0),
    name="dense-power-d12",
)
METHODS = {
    "balanced": BalancedPowerEvaluator(),
    "horner": HornerPowerEvaluator(),
    "paterson_stockmeyer": PatersonStockmeyerPowerEvaluator(baby_step=4),
}


def definition() -> dict:
    return {
        "basis": "power",
        "degree": POLYNOMIAL.degree,
        "coefficients_ascending": list(COEFFICIENTS),
        "domain": list(POLYNOMIAL.domain),
        "input": "full-slot binary64 grid on [-1,1], cyclic shift per batch member",
        "output": "the polynomial itself; no target-function approximation",
        "methods": {
            name: {
                "required_depths": evaluator.required_depths(POLYNOMIAL),
                "operations": evaluator.operation_inventory(POLYNOMIAL),
                "baby_step": 4 if name == "paterson_stockmeyer" else None,
            }
            for name, evaluator in METHODS.items()
        },
    }
