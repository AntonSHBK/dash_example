from transformers import AutoTokenizer, AutoModel
import torch
from config import CONFIG

class Embedder:
    def __init__(self, model_name=None):
        model_name = model_name or CONFIG["model_name"]
        cache_dir = CONFIG["cache_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.numpy(), texts
