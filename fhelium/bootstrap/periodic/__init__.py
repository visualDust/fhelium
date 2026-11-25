r"""Periodic reductions $r\mapsto\sin(\pi r)/\pi$ used by bootstrap EvalMod.

A reduction object, built-in or custom, provides:

- ``input_bound`` $B$ and ``fused_input_divisor`` ($B$ when CoeffsToSlots
  delivers $x=r/B$, otherwise $1$ and the input is $r$);
- ``required_depths``: depths consumed by ``evaluate``;
- ``requires_relinearization`` and optionally
  ``relinearization_depths(*, input_depth)`` for key provisioning;
- ``reference(x)``: the plaintext map on normalized $x$;
- ``evaluate(arithmetic, ciphertext, *, relinearization_key)``: return
  $\sin(\pi r)/\pi$ slot-wise at depth $d+$ ``required_depths`` through
  :class:`~fhelium.algorithms.Arithmetic`, two components, the input batch
  axes and the scale of ``arithmetic.target_scales`` at that depth.
"""

from fhelium.bootstrap.periodic._cosine import CosineDoubleAngle, CosineExpansion

__all__ = ["CosineDoubleAngle", "CosineExpansion"]
