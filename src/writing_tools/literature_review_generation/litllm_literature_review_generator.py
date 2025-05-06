from .._base import _BaseLiteratureReviewGenerator, _BaseInferenceModel
from typing import Union


VANILLA_PROMPT = """
<begin_new_scientific_abstract> 
{}
<end_new_scientific_abstract/>

<begin_other_reference_paper_abstracts>
{}
<end_other_reference_paper_abstracts/>

You are writing the related work section of a new paper. You should
do this by including ONLY the information from provided reference paper abstracts. The
section should be written as a cohesive story, identifying the strengths and weaknesses
of the reference papers and placing the new work in that context. Whenever you include
Whenever you include information from some references, you should cite them by listing them as follows: [@cite_#, @cite_#, ...], 
where # is ONLY the number of each respective reference reference. ALWAYS cite references in this way, 
do not cite them by writing things like "Reference #" or such. Do not structure the section in bullet points, but make it 
a cohesive story, written in an academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper. Do not provide any other output aside from the related work section.

IMPORTANT:
- Before the output, make sure to include the keyword: "@related_work"
"""

SENTENCE_BY_SENTENCE_PROMPT = """
<begin_new_scientific_abstract> 
{}
<end_new_scientific_abstract/>

<begin_other_reference_paper_abstracts>
{}
<end_other_reference_paper_abstracts/>

<begin_related_work_draft>
{}
<end_related_work_draft/>

You are writing the related work section of a new paper. You are writing this sentence by sentence.
You are provided with an abstract of the new paper and a raw draft of the generated work till now. 
Additionally, you will be provided with new abstracts of other reference papers ALL of which have to be cited in 
the next sentence. Your task is to write 1-2 new sentences for the related work section of the document, or
paraphrase the draft using ONLY the information in the new paper abstract and abstracts of other reference
papers. Initially, the raw draft would be empty. The section should be written as a cohesive story, 
identifying the strengths and weaknesses of the reference papers and placing the new work in that context. 
Whenever you include information from some references, you should cite them by listing them as follows: [@cite_#, @cite_#, ...], 
where # is ONLY the number of each respective reference reference. Do not  structure the section in bullet points, 
but make it a cohesive story, written in an academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper. Do not provide any other output aside from the related work section.

IMPORTANT: 
- Output the FULL related work section, including the new sentence
- Before the output, make sure to include the keyword: "@related_work"
"""


class LitLLMLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    """
    Literature review writing tool based on LitLLM

    References
    ----------
    [1] Agarwal, S., Laradji, I. H., Charlin, L., & Pal, C. (2024). Litllm: A toolkit for scientific literature review. arXiv preprint arXiv:2402.01788.
    [2] Agarwal, S., Sahu, G., Puri, A., Laradji, I. H., Dvijotham, K. D., Stanley, J., ... & Pal, C. (2024). LitLLMs, LLMs for Literature Review: Are we there yet?. Transactions on Machine Learning Research.
    """
    
    prompts = {
        "vanilla": VANILLA_PROMPT,
        "sentence": SENTENCE_BY_SENTENCE_PROMPT
    }

    def __init__(self, inference_model:_BaseInferenceModel, method:str="vanilla"):
        self.inference_model = inference_model
        self.method = method

    def predict(self, 
                query:str, 
                citations:list[str], 
                citation_ids:Union[list[int], None]=None,
                related_work_draft:Union[str, None]=None,
                citation_order:Union[list[list[int]], None]=None) -> str:

        if citation_ids is None:
            citation_ids = [i for i in range(len(citations))]

        #prompt = self.prompts[self.method]
        if self.method == "vanilla":
            related_work = self.predict_next(query, citations, citation_ids)
        elif self.method == "sentence":
            related_work = related_work_draft if related_work_draft is None else ""
            for citation_ids_next in citation_order:
                citations_next = []
                for i in range(len(citation_ids)):
                    if citation_ids[i] in citation_ids_next:
                        citations_next.append(citations[i])
                related_work = self.predict_next(query, citations_next, citation_ids_next, related_work)
            #self.prompt = prompt.format(query, references, related_work_draft)

        #prediction = self.inference_model.predict(prompt)

        #related_work = prediction.split("@related_work")[-1]

        return related_work

    def predict_next(self, 
                     query:str, 
                     citations:list[str], 
                     citation_ids:Union[list[int], None]=None,
                     related_work_draft:Union[str, None]=None):
        references = ""
        for i, citation in enumerate(citations):
            references += f"Reference {citation_ids[i]}: {citation}\n\n"

        prompt = self.prompts[self.method]
        if self.method == "vanilla":
            prompt = prompt.format(query, references)
        elif self.method == "sentence":
            self.prompt = prompt.format(query, references, related_work_draft)

        prediction = self.inference_model.predict(prompt)

        related_work = prediction.split("@related_work")[-1]

        return related_work
        