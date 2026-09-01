"""Transform and inspect Programs without assuming one operation dialect."""

from ._eliminate_dead_values import EliminateDeadValuesPass
from ._visualize_svg import (
    SvgGraphDirection,
    SvgGraphError,
    SvgGraphField,
    SvgGraphOutput,
    SvgGraphPresentation,
    SvgGraphTheme,
    SvgGraphVisualizationPass,
    SvgNodeSection,
    SvgOperationContext,
    default_svg_operation_color_key,
)

__all__ = [
    "EliminateDeadValuesPass",
    "SvgGraphDirection",
    "SvgGraphError",
    "SvgGraphField",
    "SvgGraphOutput",
    "SvgGraphPresentation",
    "SvgGraphTheme",
    "SvgGraphVisualizationPass",
    "SvgNodeSection",
    "SvgOperationContext",
    "default_svg_operation_color_key",
]
