from .._base import _BaseSuggestionGenerator, _BaseInferenceModel
import re

class SimpleLiteratureReviewGenerator(_BaseSuggestionGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, existing_text:str, new_citations:dict[str, dict[str, str]]) -> str:
        # Get all text prior to positions
        citation_context = "\n".join([new_citations[c]["abstract"] for c in new_citations.keys()])

        prompt_summary = f"""
        You will be given the abstracts of multiple research papers.
        Your goal is to summarize them briefly, in at most 50 words as if you were 
        writing a literature review for a new research paper. Make sure to write
        in an academic style. Do not mention the papers directly, but merely describe 
        the concept they are talking about. Before your output, write the keyword "Review:"

        Context: {citation_context}
        """

        # Predict based on previous text
        prediction_summary = self.inference_model.predict(prompt_summary).replace("\n", "")
        pos = list(re.finditer("Review:", prediction_summary))[-1].end()
        summary = prediction_summary[pos:]

        prompt_connect = f"""
        You will be given an unfinished literature review, and a new segment to add to it.
        Connect these two, without changing them much. Write in an academic style.
        Before your output, write the kewyword "Review:"

        Unfinished literature review: {existing_text}

        New segment: {summary}
        """
        prediction_final = self.inference_model.predict(prompt_connect).replace("\n", "")
        pos = list(re.finditer("Review:", prediction_final))[-1].end()
        out = prediction_final[pos:]

        return out