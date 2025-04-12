from .._base import _BaseInferenceModel
from ollama import chat, ChatResponse


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
        return out