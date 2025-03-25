import os

CONFIG = {
    "model_name": "bert-base-uncased",
    "pca_components": 2,
    "max_text_length": 128,
    "debug_mode": False,
    "server_host": "0.0.0.0",
    "server_port": 8050,
    "cache_dir": os.path.join(os.path.dirname(__file__), "data", "cache_dir")
}
