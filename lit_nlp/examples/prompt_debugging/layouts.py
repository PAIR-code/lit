"""Layouts for debugging language models in LIT."""

from lit_nlp.api import layout


_modules = layout.LitModuleName

LEFT_RIGHT_LAYOUT = layout.LitCanonicalLayout(
    left={
        "Examples": [_modules.DataTableModule],
        "Editor": [_modules.SingleDatapointEditorModule],
    },
    upper={  # if 'lower' not specified, this fills the right side
        "Salience": [_modules.SequenceSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(leftWidth=40),
    description="Left/right layout for language model salience.",
)

TOP_BOTTOM_LAYOUT = layout.LitCanonicalLayout(
    upper={
        "Examples": [_modules.SimpleDataTableModule],
        "Editor": [_modules.SimpleDatapointEditorModule],
    },
    lower={
        "Salience": [_modules.SequenceSalienceModule],
    },
    layoutSettings=layout.LayoutSettings(
        hideToolbar=True,
        mainHeight=40,
        centerPage=True,
    ),
    description="Simplified layout for language model salience.",
)

THREE_PANEL_LAYOUT = layout.LitCanonicalLayout(
    left={
        "Data Table": [_modules.DataTableModule],
        "Embeddings": [_modules.EmbeddingsModule],
    },
    upper={
        "Datapoint Editor": [_modules.SingleDatapointEditorModule],
        "Datapoint Generators": [_modules.GeneratorModule],
    },
    lower={
        "Salience": [_modules.SequenceSalienceModule],
        "Metrics": [_modules.MetricsModule],
    },
    layoutSettings=layout.LayoutSettings(
        mainHeight=40,
        leftWidth=40,
    ),
    description="Custom layout for language model salience.",
)

LEFT_RIGHT = "left_right"
TOP_BOTTOM = "top_bottom"
THREE_PANEL = "three_panel"

PROMPT_DEBUGGING_LAYOUTS = {
    LEFT_RIGHT: LEFT_RIGHT_LAYOUT,
    TOP_BOTTOM: TOP_BOTTOM_LAYOUT,
    THREE_PANEL: THREE_PANEL_LAYOUT,
}
