from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
from models.embedder import Embedder
from models.attention import AttentionViewer
from utils.pca import reduce_embeddings
from utils.highlight import highlight_tokens_spans
from config import CONFIG

embedder = Embedder(model_name=CONFIG["model_name"])
attention_viewer = AttentionViewer(model_name=CONFIG["model_name"])

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Визуализация эмбеддингов и Attention"),
    
    dcc.Textarea(
        id="input-texts",
        placeholder="Введите тексты через новую строку",
        style={"width": "100%", "height": 100}
    ),
    html.Button("Вычислить", id="run-btn", n_clicks=0),

    html.Hr(),

    html.Div([
        html.H4("Визуализация эмбеддингов"),
        dcc.Graph(id="embedding-plot"),
    ]),
    
    html.Hr(),

    html.Div([
        html.H4("Attention от [CLS] к токенам (1-й текст)"),
        html.Div(id="attention-highlighted", style={"lineHeight": "1.8em", "flexWrap": "wrap"}),
    ], style={"marginTop": "30px"}),

    html.Div(id="debug-output", style={"marginTop": "20px", "color": "#777"})
])

@app.callback(
    Output("embedding-plot", "figure"),
    Output("attention-highlighted", "children"),
    Output("debug-output", "children"),
    Input("run-btn", "n_clicks"),
    State("input-texts", "value")
)
def update_graph(n_clicks, raw_texts):
    if not raw_texts:
        return {}, "", "Введите хотя бы один текст"

    texts = [t.strip() for t in raw_texts.strip().split("\n") if t.strip()]
    embeddings, _ = embedder.encode(texts)

    # PCA визуализация
    if len(texts) < 2:
        fig = {}
        emb_msg = "Визуализация PCA требует минимум 2 текста"
    else:
        df = reduce_embeddings(embeddings, texts, n_components=CONFIG["pca_components"])
        fig = px.scatter(
            df,
            x="x",
            y="y",
            hover_data={"text": True},
            title="2D PCA Эмбеддинги",
            height=400
        )
        emb_msg = f"Визуализировано {len(texts)} эмбеддингов"

    # Attention визуализация (1-й текст)
    tokens, weights = attention_viewer.get_cls_attention(texts[0])
    attention_spans = highlight_tokens_spans(tokens, weights)

    return fig, attention_spans, emb_msg

if __name__ == "__main__":
    app.run(
        host=CONFIG["server_host"],
        port=CONFIG["server_port"],
        debug=CONFIG["debug_mode"]
    )
