from .._base import _BaseInferenceModel
from ollama import chat, ChatResponse
import re


class OllamaInferenceModel(_BaseInferenceModel):
    def __init__(self, model='deepseek-r1:7B'):
        self.model = model

    def predict(self, prompt:str) -> str:
        response:ChatResponse = chat(model=self.model, messages=[
            {
                "role": "user",
                "content": prompt
            }
        ])
        out = response.message.content
        if "</think>" in out:
            out = re.search("</think>(.*)", out.replace("\n", "")).group(1)
        return out