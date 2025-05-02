from .._base import _BaseInferenceModel
from ollama import chat, ChatResponse
from typing import Union, Dict, List


class OllamaInferenceModel(_BaseInferenceModel):
    def predict(self, messages:List[Dict[str, str]], **call_kwargs) -> str:
        response:ChatResponse = chat(messages=messages, **call_kwargs)
        out = response.message.content
        return out