from .._base import _BaseLiteratureReviewGenerator, _BaseInferenceModel
from typing import Union
import re

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
information from one of the references, you should cite it by writing @cite_#. Do not 
structure the section in bullet points, but make it a cohesive story, written in an 
academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper.

IMPORTANT:
- Structure your output as follows:
@related_work
[THE SECTION HERE]
"""

PLAN_BASED_PROMPT = """
<begin_new_scientific_abstract> 
{}
<end_new_scientific_abstract/>

<begin_other_reference_paper_abstracts>
{}
<end_other_reference_paper_abstracts/>

<begin_plan>
{}
<end_plan/>

You are writing the related work section of a new paper. You should
do this by including ONLY the information from provided reference paper abstracts. The
section should be written as a cohesive story, identifying the strengths and weaknesses
of the reference papers and placing the new work in that context. You are also provided a
plan mentioning the total number of lines and the citations to refer in different lines.
Whenever you include information from one of the references, you should cite it by writing @cite_#. 
Do not  structure the section in bullet points, but make it a cohesive story, written in an 
academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper.

IMPORTANT: 
- Structure your output as follows:
@related_work
[THE SECTION HERE]
"""

LEARNED_PLAN_PROMPT = """
<begin_new_scientific_abstract> 
{}
<end_new_scientific_abstract/>

<begin_other_reference_paper_abstracts>
{}
<end_other_reference_paper_abstracts/>

You are writing the related work section of a new paper. You should
do this by including ONLY the information from provided reference paper abstracts. The
section should be written as a cohesive story, identifying the strengths and weaknesses
of the reference papers and placing the new work in that context. 
Whenever you include information from one of the references, you should cite it by writing @cite_#. 
Do not  structure the section in bullet points, but make it a cohesive story, written in an 
academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper. You should first generate a plan, mentioning the total number of
lines, words and the citations to refer to in different lines. You should follow this plan when generating
sentences.

Example:

Plan: Generate the related work in [number] lines using max [number] words. Cite @cite_# on line [number]. Cite @cite_# on line [number].

IMPORTANT: 
- Structure your output as follows:
@plan
[THE PLAN HERE]

@related_work
[THE SECTION HERE]
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
the next sentence. Your task is to write 1 new sentence for the related work section of the document, or
paraphrase the draft using ONLY the information in the new paper abstract and abstracts of other reference
papers. Initially, the raw draft would be empty. The section should be written as a cohesive story, 
identifying the strengths and weaknesses of the reference papers and placing the new work in that context. 
Whenever you include information from one of the references, you should cite it by writing @cite_#. 
Do not  structure the section in bullet points, but make it a cohesive story, written in an 
academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper.

IMPORTANT: 
- Output the FULL related work section, including the new sentence
- Structure your output as follows:
@related_work
[THE SECTION HERE]
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
        "plan": PLAN_BASED_PROMPT,
        "learned_plan": LEARNED_PLAN_PROMPT,
        "sentence": SENTENCE_BY_SENTENCE_PROMPT
    }

    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model

    def predict(self, 
                query:str, 
                citations:list[str], 
                citation_ids:Union[list[int], None]=None, 
                method:str="vanilla",  
                plan:Union[str, None]=None, 
                related_work_draft:Union[str, None]=None) -> str:
        references = ""
        if citation_ids is None:
            citation_ids = [i for i in range(len(citations))]
        for i, citation in enumerate(citations):
            references += f"Reference {citation_ids[i]}: {citation}\n\n"

        prompt = self.prompts[method]
        if method == "vanilla":
            prompt = prompt.format(query, references)
        elif method == "plan":
            prompt = prompt.format(query, references, plan)
        elif method == "learned_plan":
            prompt = prompt.format(query, references)
        elif method == "sentence":
            prompt = prompt.format(query, references, related_work_draft)

        prediction = self.inference_model.predict(prompt)

        related_work = prediction.split("@related_work")[-1]
        if method == "learned_plan":
            s = prediction.replace(related_work, "")
            s = prediction.replace("@related_work", "")
            plan = s.split("@plan")[-1]
        #if method == "plan":
        #    plan = re.findall("```plan(.*)```", prediction)[-1]

        return related_work