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
        You are helping write a research paper.

        Given a few sentences from the paper, write a short sentence to continue them, by including information from the context of other papers.
        Remember: the new sentence MUST include information from the context papers.

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