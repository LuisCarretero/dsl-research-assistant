from .._base import _BaseInferenceModel
from transformers import pipeline
from typing import Union, Dict, Any, List
from huggingface_hub import InferenceClient


class HFLocalInferenceModel(_BaseInferenceModel):
    def __init__(self, **model_kwargs):
        self.pipeline = pipeline(task="text-generation", **model_kwargs)

    def _predict(self, messages:List[Dict[str, str]], **call_kwargs) -> str:
        output = self.pipeline(messages, continue_final_message=True, **call_kwargs)[0]["generated_text"][1]["content"]
        return output


class HFClientInferenceModel(_BaseInferenceModel):
    def __init__(self, **client_kwargs):
        self.client = InferenceClient(**client_kwargs)

    def _predict(self, messages:List[Dict[str, str]], **call_kwargs) -> str:
        output = self.client.chat_completion(messages, **call_kwargs).choices[0].message["content"]
        return output