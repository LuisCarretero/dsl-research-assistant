from .._base import _BaseSuggestionGenerator, _BaseInferenceModel
import re

class SimpleSuggestionGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, position_in_text:int, citations:list[str]) -> str:
        # Get all text prior to position
        citation_context = "\n".join(citations)
        text = existing_text[:position_in_text]
        prompt = f"""
        You are a tool used for giving writing suggestions to a researcher writing a paper.
        You will be given an unfinished segment of the paper.
        You will also be provided context from other papers, which you should include in the suggestion.
        Continue writing the paper by generating at most two new brief sentences.
        Your main focus should be continuing on what is written LAST in the paper so far.
        The output should match the style of the current paper.
        The output should ONLY contain the generated sentences.
        Before your output, write the keyword "Suggestion:".

        Here is the context from the other papers:

        {citation_context}

        Here is what was already written in the current paper, this is what you should continue writing on: 
        
        {text}
        """
        # Predict based on previous text
        prediction = self.inference_model.predict(prompt).replace("\n", "")
        pos = list(re.finditer("Suggestion:", prediction))[-1].end()
        out = prediction[pos:]
        return out