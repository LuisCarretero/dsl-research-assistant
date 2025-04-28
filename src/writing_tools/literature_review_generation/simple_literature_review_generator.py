from .._base import _BaseLiteratureReviewGenerator, _BaseInferenceModel
import re

class SimpleLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, query:str, citations:list[dict[int, dict[str, str]]]) -> str:
        references = ""
        c = 1
        for new_citations in citations:
            for i in new_citations:
                references += f"Reference {c}: {new_citations[i]['abstract']}\n"
                c += 1

        prompt = f"""
        New scientific paper abstract: 
        {query}

        Other reference paper abstracts:
        {references}

        You are writing the related work section of a new paper. You should
        do this by including ONLY the information from provided reference paper abstracts. The
        section should be written as a cohesive story, identifying the strengths and weaknesses
        of the reference papers and placing the new work in that context. Whenever you include
        information from one of the references, you should cite it by writing (@cite_#). Do not 
        structure the section in bullet points, but make it a cohesive story, written in an 
        academic style. Do not provide references at the end. Do not copy the abstracts of the 
        reference papers directly, but consisely compare and constrast them to the main work. Do not
        reference the new abstract paper.
        
        IMPORTANT: Before the output, make sure to write the keyword "@related_work:"
        """
        prediction = self.inference_model.predict(prompt)
        out = prediction.split("@related_work:")[-1]
        return out