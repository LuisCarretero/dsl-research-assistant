from .._base import _BaseSuggestionGenerator, _BaseInferenceModel
import re

class SimpleSuggestionGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, position_in_text:int, citations:list[str]) -> str:
        # Get all text prior to position
        citation_context = "\n".join(citations)
        previous_sentences = existing_text[:position_in_text].split(".")
        text = ".".join(previous_sentences[min(10, len(previous_sentences))*-1:]) # Fetch the last 10 sentences

        prompt = f"""
        You are a researcher writing a paper. Continue writing it given the following text: 

        [BEGINING OF TEXT TO CONTINUE WRITING]
        {text}
        [END OF TEXT TO CONTINUE WRITING]

        Do this by writing a single short sentence which includes the following context:

        [BEGINING OF CONTEXT TO BE INCLUDED]
        {citation_context}
        [END OF CONTEXT TO BE INCLUDED]

        Keep in mind, the context comes from a different paper, so it is NOT the contribution 
        of the paper you are writing, but is merely related to it.

        Your output MUST be short, and contain only one sentence.
        Before your output, write the keyword "Suggestion:".
        """

        # Predict based on previous text
        prediction = self.inference_model.predict(prompt).replace("\n", "")
        pos = list(re.finditer("Suggestion:", prediction))[-1].end()
        out = prediction[pos:]
        return out