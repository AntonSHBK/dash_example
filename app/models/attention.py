from transformers import AutoTokenizer, AutoModel
import torch
from config import CONFIG

class AttentionViewer:
    def __init__(self, model_name=None):
        model_name = model_name or CONFIG["model_name"]
        cache_dir = CONFIG["cache_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager",
            cache_dir=cache_dir
        )
        self.model.eval()

    def get_cls_attention(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attentions = outputs.attentions
        last_layer = attentions[-1][0]
        cls_attention = last_layer.mean(dim=0)[0]
        return tokens, cls_attention.numpy()
