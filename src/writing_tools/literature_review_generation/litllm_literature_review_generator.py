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

    def replace_ref_nums(self, prediction:str, reference_ids:list[int]) -> str:
        # Find the references and replace each of them with the correct number
        in_text_ref_strings = re.findall(r"\[[0-9, -]\]", prediction)
        for i, in_text_ref_string in enumerate(in_text_ref_strings):
            # Look for range reference types
            range_refs = re.findall(r"[0-9]+-[0-9]+", in_text_ref_string)
            for range_ref in range_refs:
                minus_pos = range_ref.find("-")
                a = int(range_ref[:minus_pos])
                b = int(range_ref[minus_pos+1:])
                in_text_ref_string = in_text_ref_string.replace(range_ref, str([i for i in range(a, b+1)]).replace("[", "").replace("]", ""))
            # Try parsing the list and replacing the reference IDs if possible
            try:
                ref_nums = ast.literal_eval(in_text_ref_string) # Parse the string to lists of integers
                for j, ref_wrong_num in enumerate(ref_nums):
                    try:
                        # Try converting to a integer
                        ref_wrong_num = int(ref_wrong_num)
                        # Fetch the true reference number if possible
                        if ref_wrong_num > 0 and ref_wrong_num <= len(reference_ids):
                            ref_nums[j] = reference_ids[ref_wrong_num-1]
                    except:
                        continue
                # Replace the text with the modified list
                prediction = prediction.replace(in_text_ref_string, str(ref_nums))
            except:
                continue
        return prediction
    
    def create_references_text(self, references, reference_ids):
        references_text = ""
        for i, reference in enumerate(references):
            references_text += f"Reference {reference_ids[i]}: {reference}"
            if i < len(references)-1:
                references_text += "\n\n"

    def predict_entire(self, query:str, references:list[str], reference_ids:list[int]):
        references_text = self.create_references_text(references, reference_ids)
        prompt = ENTIRE_PROMPT.format(query, references_text)
        prediction = self.inference_model.predict(prompt).strip("\n ")
        #prediction = self.replace_ref_nums(prediction, reference_ids)
        return prediction
    
    def predict_next(self, query:str, references:list[str], reference_ids:list[int], related_work_draft:str):
        references_text = self.create_references_text(references, reference_ids)
        prompt = NEXT_PROMPT.format(query, references_text, related_work_draft)
        prediction = self.inference_model.predict(prompt).strip("\n ")
        #prediction = self.replace_ref_nums(prediction, reference_ids)
        return prediction

    """
    def predict_(self, 
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
            if citation_order is None:
                citation_order = [[i] for i in citation_ids]
            related_work = related_work_draft if related_work_draft is not None else ""
            for citation_ids_next in tqdm(citation_order, desc="Generating Related Work section..."):
                citations_next = []
                for i in range(len(citation_ids)):
                    if citation_ids[i] in citation_ids_next:
                        citations_next.append(citations[i])
                related_work += self.predict_next(query, citations_next, citation_ids_next, related_work)
                print(related_work)
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
            references += f"Reference @cite_{citation_ids[i]}: {citation}"
            if i < len(citations)-1:
                references += "\n\n"

        prompt = self.prompts[self.method]
        if self.method == "vanilla":
            prompt = prompt.format(query, references)
        elif self.method == "sentence":
            prompt = prompt.format(query, references, related_work_draft)

        prediction = self.inference_model.predict(prompt)
        prediction = prediction.strip("\n ")

        return prediction
    """
        