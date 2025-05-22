from openai import OpenAI
from .._base import _BaseInferenceModel
from typing import List, Dict


class OpenAIInferenceModel(_BaseInferenceModel):
    def __init__(self, **client_kwargs):   
        self.client = OpenAI(**client_kwargs)

    def _predict(self, messages:List[Dict[str, str]], **call_kwargs) -> str:
        output = self.client.chat.completions.create(messages, **call_kwargs).choices[0].message["content"]
        return output