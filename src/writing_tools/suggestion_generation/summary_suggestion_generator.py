from .._base import _BaseSuggestionGenerator, _BaseInferenceModel
import re

class SummarySuggestionGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, position_in_text:int, citations:list[str]) -> str:
        # Get all text prior to position
        citation_context = "\n".join(citations)
        previous_sentences = existing_text[:position_in_text].split(".")
        text = ".".join(previous_sentences[min(10, len(previous_sentences))*-1:]) # Fetch the last 10 sentences

        prompt_summary = f"""
        Summarize these papers:

        [Beginning of paper abstracts]
        {citation_context}
        [End of paper abstracts]

        Keep your answer short. At most one sentence.
        Before your output, write the keyword "Summary:".
        """

        summary = self.inference_model.predict(prompt_summary).replace("\n", "")
        pos = list(re.finditer("Summary:", summary))[-1].end()
        summary = summary[pos:]

        prompt_suggestion = f"""
        Continue the following text, as if you were writing a research paper: 

        {text}

        Do this by writing a single short sentence which includes the following context:

        {summary}

        Your output MUST be short, and contain only one sentence.
        Before your output, write the keyword "Suggestion:".
        """

        # Predict based on previous text
        prediction = self.inference_model.predict(prompt_suggestion).replace("\n", "")
        pos = list(re.finditer("Suggestion:", prediction))[-1].end()
        out = prediction[pos:]
        return out