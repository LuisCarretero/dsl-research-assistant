from fastapi import APIRouter
from src.api.models import TextRequest, ContinuationsResponse, Continuation
from writing_tools import LitLLMLiteratureReviewGenerator, HFClientInferenceModel
from dotenv import load_dotenv
import os
import re
import pandas as pd

load_dotenv()

DB_SUPERDIR = os.environ.get("DB_SUPERDIR")
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

    text_old = text

    text = text.strip("\n ")

    # Get the abstract and related work section
    abstract_title = "[Abstract]"
    related_work_title = "[Related Work]"

    abstract = text[text.find(related_work_title)].replace(abstract_title, "").strip("\n ")
    related_work = text[text.find(related_work_title)+len(related_work_title):].strip("\n ")

    new_refs_string = re.findall(r"\<refs\>(.*)\</refs\>", related_work)

    if len(new_refs_string) == 0:
        return ContinuationsResponse(continuations=[
            Continuation(
                id=1,
                text=text_old + "ERROR: References must be contained inside: <refs> </refs>",
                confidence=0
            )
        ])
    else:
        new_refs_string = new_refs_string[0]

    # Remove the new reference string from the related work
    related_work = related_work.replace(new_refs_string, "").strip("\n ")

    # Remove the <LLM> and <ERROR> citations
    related_work = related_work.replace("<LLM>", "").replace("<ERROR>", "")

    # Get all the old references
    try:
        old_refs = re.findall(r"\[([^\]]*)\]", related_work)
    except:
        old_refs = []

    # Get all the new references
    try:
        new_refs = re.findall(r"\[([^\]]*)\]", new_refs_string)
    except:
        new_refs = []
    
    # Concatenate all the references
    all_refs = old_refs + new_refs

    unique_refs = list(set(all_refs))

    # Fetch the reference data
    reference_df = pd.read_parquet(os.path.join(DB_SUPERDIR, "documents.parquet"))

    # Extract the reference numbers and reference abstracts
    new_ref_nums = []
    new_ref_abstracts = []
    related_work_draft = related_work
    for i, ref in enumerate(unique_refs):
        # Make sure to replace the citations with numbers
        related_work_draft = related_work.replace(ref, str(i))
        # If it is a new citation, add the abstract and reference number
        if ref in new_refs:
            row = reference_df.loc[reference_df["cit_str"] == ref.strip(), "abstract"]
            if len(row) > 0:
                new_ref_abstracts.append(row.values[0])
                new_ref_nums.append(i+1)
            
    if len(new_ref_nums) == 0:
        return ContinuationsResponse(continuations=[
            Continuation(
                id=1,
                text=text_old + "ERROR: invalid or non-existant references were given",
                confidence=100
            )
        ])

    prediction = model.predict_next(abstract, 
                                    new_ref_abstracts, 
                                    reference_ids=new_ref_nums,
                                    related_work_draft=related_work_draft)
    
    # Insert the original references back into the predictions
    all_predicted_ref_list_strings = re.findall(r"\[[^\]]*\]", prediction)
    for predicted_ref_list_string in all_predicted_ref_list_strings:
        # First, check if it has range references to replace them
        range_refs = re.findall(r"[0-9]+-[0-9]+", predicted_ref_list_string)
        ref_cleaned = predicted_ref_list_string
        for range_ref in range_refs:
            num1 = int(range_ref[:range_ref.find("-")])
            num2 = int(range_ref[range_ref.find("-")+1:])
            replacement_range_str = ", ".join([i for i in range(num1, num2+1)])
            ref_cleaned = ref_cleaned.replace(range_ref, replacement_range_str)
        # Next, convert to a list:
        predicted_ref_list = predicted_ref_list_string[1:-1].split(",")
        for i, ref in enumerate(predicted_ref_list):
            # Try converting to integer
            try:
                ref = int(ref)
                # If the reference exists somewhere in the text, insert it
                if ref >= 1 and ref <= len(unique_refs):
                    predicted_ref_list[i] = f"[{unique_refs[ref-1]}]"
                # Otherwise, indicate halucination by writing <LLM>
                else:
                    predicted_ref_list[i] = f"<LLM>"
            # If it is not possible to convert to an integer, write <ERROR>
            except:
                predicted_ref_list[i] = "<ERROR>"
        # Replace the LLM-predicted references with the modified ones
        prediction = prediction.replace(predicted_ref_list_string, " ".join(predicted_ref_list))


    continuations = [
        Continuation(
            id=1,
            text=text+" "+prediction,
            confidence=90
        )
    ]
    
    return ContinuationsResponse(continuations=continuations)