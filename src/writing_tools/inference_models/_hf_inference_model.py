from .._base import _BaseInferenceModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import numpy as np
from typing import Union
from huggingface_hub import InferenceClient


class HFLocalInferenceModel(_BaseInferenceModel):
    def __init__(self, **model_kwargs):
        self.pipeline = pipeline(task="text-generation", **model_kwargs)

    def predict(self, user_prompt:str, system_prompt:Union[str,None]=None, **call_kwargs) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        output = self.pipeline(messages, **call_kwargs)[0]["generated_text"][1]["content"]
        return output


class HFClientInferenceModel(_BaseInferenceModel):
    def __init__(self, **client_kwargs):
        self.client = InferenceClient(**client_kwargs)

    def predict(self, user_prompt:str, system_prompt:Union[str,None]=None, **call_kwargs) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        output = self.client.chat_completion(messages, **call_kwargs).choices[0].message["content"]
        return output