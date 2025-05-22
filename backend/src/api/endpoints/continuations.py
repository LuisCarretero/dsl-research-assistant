from fastapi import APIRouter
from src.api.models import TextRequest, ContinuationsResponse, Continuation
from writing_tools import LitLLMLiteratureReviewGenerator, HFClientInferenceModel
from dotenv import load_dotenv
import os
import re
import pandas as pd
import ast

HF_TOKEN = os.environ.get("HF_TOKEN")

llm = HFClientInferenceModel(
    provider = "nebius",
    api_key = HF_TOKEN
)
llm.set_default_call_kwargs(
    model = "meta-llama/Llama-3.1-70B-Instruct"
)

model = LitLLMLiteratureReviewGenerator()

router = APIRouter()

@router.post("/generate-continuation/", response_model=ContinuationsResponse)
async def generate_continuation(request: TextRequest):
    """
    Generate text continuations based on the provided text.
    """
    text = request.text

    # Get the abstract and related work section
    abstract_title = "[Abstract]"
    related_work_title = "[Related Work]"

    abstract = text[text.find(related_work_title)].replace(abstract_title, "").strip("\n ")
    related_work = text[text.find(related_work_title)+len(related_work_title):].strip("\n ")

    try:
        all_citations = re.findall(r"\[(.*)\]", related_work)
        last_citation = [all_citations[-1]]
    except:
        all_citations = []
        last_citation = []

    unique_citations = list(set(all_citations))

    reference_df = pd.read_parquet(...) # TODO: add directory

    reference_nums = []
    reference_abstracts = []
    related_work_draft = ""
    for i, citation in enumerate(unique_citations):
        # Make sure to replace the citations with numbers
        related_work_draft = related_work.replace(citation, str(i))
        # If it is a new citation, add the abstract and reference number
        if citation in last_citation:
            reference_abstracts.append(reference_df.loc[reference_df["cit_str"] == citation, "abstract"].values[0])
            reference_nums.append(i+1)

    prediction = model.predict_next(abstract, 
                                    reference_abstracts, 
                                    reference_ids=reference_nums,
                                    related_work_draft=related_work_draft)
    
    # Insert the original references back into the predictions
    all_predicted_citations = re.findall(r"\[[^\]]*\]", prediction)
    valid_predicted_citations = re.findall(r"\[[0-9, -]+\]", prediction)
    # Remove the invalid citations
    for citation in all_predicted_citations:
        if citation not in valid_predicted_citations:
            prediction = prediction.replace(citation, "<CITING_ERROR>")
    for citation in valid_predicted_citations:
        # First, check if it has range citations
        range_cits = re.findall(r"[0-9]+-[0-9]+", citation)
        citation_cleaned = citation
        for range_cit in range_cits:
            num1 = int(range_cit[:range_cit.find("-")])
            num2 = int(range_cit[range_cit.find("-")+1:])
            replacement_range_str = ", ".join([i for i in range(num1, num2+1)])
            citation_cleaned = citation_cleaned.replace(range_cit, replacement_range_str)
        try:
            citation_nums = [int(i) for i in ast.literal_eval(citation_cleaned)]
            for i, cit_num in enumerate(citation_nums):
                if cit_num >= 1 and cit_num <= len(unique_citations):
                    citation_nums[i] = unique_citations[cit_num-1]
                else:
                    citation_nums[i] = "LLM"
            replacement_str = str(citation_nums)
        except:
            replacement_str = "<CITING_ERROR>"
        prediction = prediction.replace(citation, replacement_str)


    continuations = [
        Continuation(
            id=1,
            text=text+" "+prediction,
            confidence=90
        )
    ]
    """
    continuations = [
        Continuation(
            id=1,
            text=text + " Furthermore, the implementation of neural networks in this context provides a robust framework for analyzing complex patterns within the dataset.",
            confidence=95
        ),
        Continuation(
            id=2,
            text=text + " This approach, however, is not without limitations. Several researchers have pointed out potential biases that may emerge from such methodologies.",
            confidence=90
        ),
        Continuation(
            id=3,
            text=text + " To address these challenges, we propose a novel algorithm that combines the strengths of both supervised and unsupervised learning techniques.",
            confidence=85
        )
    ]
    """
    
    return ContinuationsResponse(continuations=continuations)