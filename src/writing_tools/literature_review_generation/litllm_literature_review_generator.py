from .._base import _BaseLiteratureReviewGenerator, _BaseInferenceModel
from typing import Union
from tqdm import tqdm


VANILLA_PROMPT = """
You are a helpful research assistant who is helping with literature review of a research idea.

You will be provided with an abstract of a scientific document and other references papers in triple quotes. 
Your task is to write the related work section of the document using only the provided abstracts and other references papers. 
Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing 
the strengths and weaknesses while also motivating the proposed approach. You should cite the other related documents as 
[#] whenever you are referring it in the related work (here, # is the given number of the reference). Do not write it as Reference #. Do not cite abstract. Do not include 
any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and 
contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. 
Please cite all the provided reference papers. Provide the output in maximum 1000 words.

```
Main abstract: {}

{}
``` Related Work:
"""

PLAN_PROMPT = """
You are a helpful research assistant who is helping with literature review of a research idea.

You will be provided with an abstract of a scientific document and other references papers in triple quotes. 
Your task is to write the related work section of the document using only the provided abstracts and other references papers. 
Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work comparing 
the strengths and weaknesses while also motivating the proposed approach. Additionally, you will be given a plan specifying when to cite
which reference, and which references to cite together. You should cite the other related documents as 
[#] whenever you are referring it in the related work (here, # is the given number of the reference). Do not write it as Reference #. Do not cite abstract. Do not include 
any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and 
contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. 
Please cite all the provided reference papers. Provide the output in maximum 1000 words.

```
Main abstract: {}

{}

Plan: {}
``` Related Work:
"""

"""
EXAMPLE PLAN:

Plan: Cite the references in this order
[1], [2], [15] should be cited together
[4] should be cited before [8]
"""

SENTENCE_PROMPT3 = """
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

```
Main abstract: {}

{}

Unfinished related work: {}
``` New sentence(s) in the Related Work:
"""

SENTENCE_BY_SENTENCE_PROMPT = """
<begin_new_scientific_abstract> 
{}
</end_new_scientific_abstract>

<begin_other_reference_paper_abstracts>
{}
</end_other_reference_paper_abstracts>

<begin_related_work_draft>
{}
</end_related_work_draft>

You are writing the related work section of a new paper. You are writing this sentence by sentence.
You are provided with an abstract of the new paper and a raw draft of the generated work till now. 
Additionally, you will be provided with new abstracts of other reference papers ALL of which have to be cited in 
the next sentence. Your task is to write a new sentences for the related work section of the document,
using ONLY the information in the new paper abstract and abstracts of other reference
papers. Initially, the raw draft would be empty. The section should be written as a cohesive story, 
identifying the strengths and weaknesses of the reference papers and placing the new work in that context. 
Whenever you include information from some references, you should cite them by listing them as follows: [@cite_#, @cite_#, ...], 
where # is ONLY the number of each respective reference reference. ALWAYS cite references in this way, 
do not cite them by writing things like "Reference #" or such. Do not structure the section in bullet points, 
but make it a cohesive story, written in an academic style. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and constrast them to the main work. Do not
reference the new abstract paper. Do not provide any other output aside from the related work section.
Do not output the title of the section, only output the section itself.

IMPORTANT:
Output only the new generated sentence, with the in text citations.
When citing a reference, cite it exactly with the format as specified above. 
Example of citing:

[@cite_1, @cite_32]

In this example, two references, namely 1 and 32 are cited

Another example of citing:

[@cite_16]

In this example, only one reference, namely 16 is cited.
"""
#IMPORTANT: 
#- Output only the newly generated sentence(s)
#- Before the output, make sure to include the keyword: "@related_work"
#"""

SENTENCE_PROMPT = """
You are writing a "Related Work" section of a new research paper. You are given the abstract
of the new paper, the Related Work written so far, and the abstracts of some references.
Your goal is to write one or two brief sentences, so that they continue on the so far written Related Work. 
The sentence(s) should naturally continue on what is written in the Related Work so far, without modifying 
what has been written previously.
The information in the new sentence(s) should be based ONLY on what is provided in the reference abstracts. 
If the provided Related Work is empty, then the new sentence(s) should be the first sentences of the Related Work. 
You are writing a Related Work section, which means that you should identify the strengths and weaknesses of the 
provided literature, and when deemed fit specify how the new research contributes to the field. The section should
have a cohesive storyline touching those topics. Keep in mind: you should include information that concerns ALL 
the provided references. Be specific, write about the topics that concern the new papers explicitly and identify
how the referenced abstracts relate to them.

Whenever you include information from one or more of the references, you should cite them in a list by writing
[@cite_#, @cite_#, ...], where # is the number of the reference you are citing (which is given to you). 
For example, writing [@cite_4, @cite_15] means you are citing references 4 and 15.

Do not provide your output in terms of bullet points. Do not provide references at the end. Do not copy the abstracts of the 
reference papers directly, but consisely compare and contrast them to the main work. Do not come up with your own information,
only use the information provided in the reference abstracts. Do not reference the new abstract paper. Do not provide any other 
output aside from the Related Work section itself, without any explanations of why you wrote what you did. Do not output any 
extra whitespaces or tags. Do not output the title of the section, only output the section itself. In your final answer, 
provide the entire section that has been written so far, together with the newly written sentence(s).

```
[New paper abstract]
{}

[Reference abstracts]
{}

[Unfinished Related Work]
{}
```

Related Work:
"""

SENTENCE_PROMPT2 = """
You are helping write the "Related Work" section of a research paper.

You will be given the abstract of the current paper, the abstract of a few reference papers and an unfinished "Related Work" section in triple quotes.
Please continue the next 1-2 sentences of the 'Related Work' section only. If the 'Related Work' section written so far is empty, start writing it.
You must include information from the reference abstracts in the new sentence. Make sure that the sentence flows naturally
with the 'Related Work written so far'. Before you write the new sentences, copy the entire related work written so far EXACTLY, then write the new sentences. 
Do not provide any other output/explanation or whitespaces. When you use information from one or more of the references, you MUST cite them as [@cite_#, @cite_#, ...].
Do not refer to the references directly in any other way except that

```
[Current paper abstract]
{}

[Reference paper abstracts]
{}

[Unfinished Related Work section]
{}
```

Related Work:
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
        "sentence": SENTENCE_PROMPT3
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
        