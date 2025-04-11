from .._base import _BaseSuggestionGenerator, _BaseInferenceModel

class SimpleSuggestionGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, position_in_text:int) -> str:
        # Get all text prior to position
        text = existing_text[:position_in_text]
        prompt = f"""
        You are researcher writing a paper. 
        Generate at most two sentences following the text that has already been written.
        The output should match the style of the given text.
        Here is what was already written: {text}
        """
        # Predict based on previous text
        prediction = self.inference_model.predict(prompt)
        return prediction