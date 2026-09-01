"""Independent semantic-state axes shared by CKKS values.

``PlaintextRepresentation`` states what the payload means. Polynomial domain,
modulus basis, and residue representation are independent axes for RNS values;
none can be inferred from ``level`` alone.
"""

from typing import Literal

PlaintextRepresentation = Literal[
    "slots",
    "integer_coefficients",
    "approximate_coefficients",
    "rns",
]
# ``coefficient`` indexes polynomial coefficients; ``ntt`` indexes evaluations.
PolynomialDomain = Literal["coefficient", "ntt"]
# ``Q`` is the active ordinary basis; ``QP`` appends special key-switch rows.
ModulusBasis = Literal["Q", "QP"]
# ``standard`` stores ordinary residues; ``montgomery`` stores Montgomery form.
ResidueRepresentation = Literal["standard", "montgomery"]
# Physical expansion of an operation-ready encoded axis. This is not a
# user-visible CKKS slot-packing layout.
CompressedPlaintextLayout = Literal["cyclic", "contiguous", "strided_sparse"]
