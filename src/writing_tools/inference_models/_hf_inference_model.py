from .._base import _BaseInferenceModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import numpy as np


class HFInferenceModel(_BaseInferenceModel):
    def __init__(self, model:str, tokenizer:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=device, max_new_tokens=np.inf)

    def predict(self, prompt: str) -> str:
        output = self.pipeline([{"role": "user", "content": prompt}])[0]["generated_text"][1]["content"]
        return output