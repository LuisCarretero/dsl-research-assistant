from .._base import _BaseSuggestionGenerator, _BaseInferenceModel
import re

class SimpleSuggestionGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, position_in_text:int) -> str:
        # Get all text prior to position
        text = existing_text[:position_in_text]
        prompt = f"""
        You are researcher writing a paper. 
        You will be given an unfinished segment of the paper.
        Continue writing the paper by generating at most two new brief sentences.
        Your main focus should be continuing on what is written LAST.
        The output should match the style of the given text.
        The output should ONLY contain the generated sentences.
        Before your output, you should always write the keyword "Suggestion:".
        Here is what was already written: 
        
        {text}
        """
        # Predict based on previous text
        prediction = self.inference_model.predict(prompt).replace("\n", "")
        pos = list(re.finditer("Suggestion:", prediction))[-1].end()
        out = prediction[pos:]
        return out