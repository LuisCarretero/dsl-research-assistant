from .._base import _BaseLiteratureReviewGenerator, _BaseInferenceModel
from typing import Union
from tqdm import tqdm
import re
import ast


ENTIRE_PROMPT = """
You are a helpful research assistant who is helping with literature review of a research idea.

You will be provided with an abstract of a scientific document and other references papers in triple quotes. 
Your task is to write the related work section of the document using only the provided abstracts and other references papers. 
Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing 
the strengths and weaknesses while also motivating the proposed approach. You should cite the other related documents as 
[#] whenever you are referring it in the related work (here, # is the given number of the reference). Do not write it as Reference #. Do not cite abstract. Do not include 
any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and 
contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. 
Please cite all the provided reference papers. Provide the output in maximum 1000 words.

Examples of how you should cite the references:
Example 1, citing Reference 1: [1],
Example 2, citing References 4 and 6 in the same place: [4, 6]

```
Main abstract: {}

{}
``` Related Work:
"""

NEXT_PROMPT = """
You are a helpful research assistant who is helping with literature review of a research idea.

You will be provided with an abstract of a scientific document and other references papers in triple quotes. 
Your task is to write write one or two new sentences of a related work section of the document using only the provided abstracts and other references papers. 
Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing 
the strengths and weaknesses while also motivating the proposed approach. Additionally, you will be given an unfinished version 
of the Related Work, which you should continue writing (it might also be empty, in which case you should start writing it). Your output should contain ONLY the new sentences. 
You should cite the other related documents as [#] whenever you are referring it in the related work (here, # is the given number of the reference). Do not write it as Reference #. 
The new sentence(s) should cover a topic common to the references, so you should cite them together. Do not cite abstract. Do not include any extra notes or newline 
characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. 
Do not provide references at the end. Please cite all the provided reference papers together. Provide the output in maximum 80 words.

Examples of how you should cite the references:
Example 1, citing Reference 1: [1],
Example 2, citing References 4 and 6 in the same place: [4, 6]

```
Main abstract: {}

{}

Unfinished related work: {}
``` New sentence(s) in the Related Work:
"""


class LitLLMLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    """
    Literature review writing tool based on LitLLM

    ## References
     
    - [1] Agarwal, S., Laradji, I. H., Charlin, L., & Pal, C. (2024). Litllm: A toolkit for scientific literature review. arXiv preprint arXiv:2402.01788.
    - [2] Agarwal, S., Sahu, G., Puri, A., Laradji, I. H., Dvijotham, K. D., Stanley, J., ... & Pal, C. (2024). LitLLMs, LLMs for Literature Review: Are we there yet?. Transactions on Machine Learning Research.
    """
    def __init__(self, inference_model:_BaseInferenceModel):
        self.inference_model = inference_model
    
    def create_references_text(self, references, reference_ids):
        references_text = ""
        for i, reference in enumerate(references):
            references_text += f"Reference {reference_ids[i]}: {reference}"
            if i < len(references)-1:
                references_text += "\n\n"
        return references_text

    def predict_entire(self, query:str, references:list[str], reference_ids:list[int]):
        references_text = self.create_references_text(references, reference_ids)
        prompt = ENTIRE_PROMPT.format(query, references_text)
        prediction = self.inference_model.predict(prompt).strip("\n ")
        return prediction
    
    def predict_next(self, query:str, references:list[str], reference_ids:list[int], related_work_draft:str):
        references_text = self.create_references_text(references, reference_ids)
        prompt = NEXT_PROMPT.format(query, references_text, related_work_draft)
        prediction = self.inference_model.predict(prompt).strip("\n ")
        return prediction

        