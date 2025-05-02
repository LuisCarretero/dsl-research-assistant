from openai import OpenAI
from .._base import _BaseInferenceModel
from typing import Union


class OpenAIInferenceModel(_BaseInferenceModel):
    def __init__(self, **client_kwargs):   
        self.client = OpenAI(**client_kwargs)

    def predict(self, user_prompt:str, system_prompt:Union[str, None]=None, **call_kwargs) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        output = self.client.chat.completions.create(messages, **call_kwargs).choices[0].message["content"]
        return output