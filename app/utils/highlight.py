from dash import html
import matplotlib
from matplotlib import cm

def highlight_tokens_spans(tokens, weights, cmap="Reds"):
    norm = matplotlib.colors.Normalize(vmin=min(weights), vmax=max(weights))
    colormap = cm.get_cmap(cmap)

    spans = []
    for token, weight in zip(tokens, weights):
        color = matplotlib.colors.rgb2hex(colormap(norm(weight)))
        span = html.Span(
            token + " ",
            style={
                "backgroundColor": color,
                "padding": "2px",
                "margin": "1px",
                "borderRadius": "4px",
                "display": "inline-block",
                "fontFamily": "monospace"
            }
        )
        spans.append(span)
    return spans
