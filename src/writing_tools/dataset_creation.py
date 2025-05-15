import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import re
import ast
from pyalex import Works
import pandas as pd
import pprint
from writing_tools import LitLLMLiteratureReviewGenerator, HFClientInferenceModel, LexRankLiteratureReviewGenerator
from writing_tools._base import _BaseLiteratureReviewGenerator
from rouge_score import rouge_scorer
from evaluate import load
from typing import Union
import numpy as np

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

def create_related_work_dataset():
    papers_path = os.path.join(DATA_DIR, "challenge10_batch_1\\CVPR_2024\\Conversions\\opencvf-data\\md")
    reference_data_path = os.path.join(DATA_DIR, "reference_data")

    # Fetch the originial data info
    orig_df = pd.read_csv(os.path.join(reference_data_path, "orig.csv"))
    orig_df["rw_refs_oaids"] = None

    rw_dataset = [] 
    # Format
    # {
    #   "title": "string",
    #   "abstract": "string",
    #   "related_work": "string",
    #   "rw_in_text_ref_nums": ["integer"]
    #   "rw_in_text_ref_abstracts": ["string"]
    # }

    #print(orig_df["ss_refs_doi"])

    # Fetch the reference data info
    refs_df = pd.read_csv(os.path.join(reference_data_path, "refs.csv"))

    count_used = 0
    count_total = 0
    # Loop through all original papers 
    for paper in tqdm(os.listdir(papers_path)):
        orig_paper = orig_df[orig_df["fname"] == paper.replace(".md", ".txt")]
        count_total += 1
        # Read the paper text
        with open(os.path.join(papers_path, paper), "r", encoding="utf-8") as p:
            paper_text = p.read()
        # Find the sections
        sections = paper_text.split("##")
        # Find related work section title
        related_work_titles = re.findall(r"## ([0-9]. Related Work)", paper_text)
        # Find the references section title
        references_titles = re.findall(r"## (References)", paper_text)
        # Find the abstract section title
        abstract_titles = re.findall(r"## (Abstract)", paper_text)
        
        if len(related_work_titles) > 0 and len(references_titles) > 0 and len(abstract_titles) > 0 and len(orig_paper) > 0:
            related_work_title = related_work_titles[0]
            references_title = references_titles[0]
            abstract_title = abstract_titles[0]
            # Find the appropriate section
            related_work_section = ""
            references_section = ""
            abstract_section = ""
            #print(sections)
            for section in sections:
                if related_work_title in section:
                    related_work_section = section
                elif references_title in section:
                    references_section = section
                elif abstract_title in section:
                    abstract_section = section
            if related_work_section.strip("\n ") == "" or \
                abstract_section.strip("\n ") == "" or \
                references_section.strip("\n ") == "":
                continue
            dict_new = {
                "oaid": orig_paper["oaid"].values[0],
                "title": orig_paper["title"].values[0],
                "abstract": clean_section(abstract_section),
                "related_work": clean_section(related_work_section),
                "rw_in_text_ref_nums": [],
                "rw_in_text_ref_abstracts": []
            }
            # Find all the references in the related works
            related_work_in_text_citations = re.findall(r"\[[0-9, ]+\]", related_work_section)
            in_text_citations = []
            for elem in related_work_in_text_citations:
                in_text_citations += ast.literal_eval(elem)
            # From the references, remove the ones that are not in the related work
            references_list = references_section.split("\n")
            related_work_oaids = []
            related_work_ref_nums = []
            related_work_ref_abstracts = []
            l = orig_paper["refs_oaids_from_dois"].tolist()
            rw_oaids = ast.literal_eval(l[0]) if len(l) > 0 else []
            rw_oaids = [i.upper() for i in rw_oaids]
            rw_info = refs_df.loc[refs_df["oaid"].isin(rw_oaids)]
            for reference in references_list:
                ref_num = re.findall(r"\[([0-9]+)\]", reference)
                if len(ref_num) > 0:
                    ref_num = int(ref_num[0])#[0]
                    #ref_num = ast.literal_eval(ref_num)[0]
                    
                    if ref_num in in_text_citations:
                        for i in rw_info.index:
                            ref_abstract = rw_info.loc[i, "abstract"]
                            ref_title = rw_info.loc[i, "title"]

                            if isinstance(ref_title, str) and ref_title.lower() in reference.lower() and \
                                isinstance(ref_abstract, str) and ref_abstract != "":
                                related_work_oaids.append(rw_info.loc[i, "oaid"])
                                related_work_ref_nums.append(ref_num)
                                related_work_ref_abstracts.append(ref_abstract)
                                break
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_oaids"] = str(related_work_oaids)
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_in_text_num"] = str(related_work_ref_nums)
            dict_new["rw_in_text_ref_abstracts"] = related_work_ref_abstracts
            dict_new["rw_in_text_ref_nums"] = related_work_ref_nums
            if len(related_work_oaids) > 0:
                rw_dataset.append(dict_new)
                count_used += 1
        else:
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_oaids"] = "[]"
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_in_text_num"] = "[]"
    print(f"Used {count_used}/{count_total} = {count_used/count_total*100}% of papers")
    with open(os.path.join(reference_data_path, "rw_dataset.json"), "w") as f:
        json.dump(rw_dataset, f)
    orig_df.to_csv(os.path.join(reference_data_path, "new_orig.csv"), index=False)


def extract_citation_order(paper_ref_dict:dict):
    related_work = paper_ref_dict["related_work"]
    citations = re.findall(r"\[[0-9, ]+\]", related_work)
    in_text_ref_dict_order = []
    for citation in citations:
        temp = ast.literal_eval(citation)
        in_text_ref_dict_order_i = []
        for i in temp:
            if int(i) in paper_ref_dict["rw_in_text_ref_nums"]:
                in_text_ref_dict_order_i.append(int(i))
        #in_text_ref_dict_order_i = [int(i) for i in in_text_ref_dict_order_i]
        in_text_ref_dict_order.append(in_text_ref_dict_order_i)
    return in_text_ref_dict_order


def clean_section(section:str):
    # Strip white spaces and newlines
    section = section.strip("\n ")
    # Remove images
    section = section.replace("<!-- image -->\n\n", "")
    paragraphs = section.split("\n\n")
    # Remove the title
    section = section.replace(paragraphs[0]+"\n\n", "")
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith("Figure ") or paragraph.startswith("Table ") or (paragraph.startswith("|") and paragraph.endswith("|")):
            if i < len(paragraphs)-1:
                section = section.replace(paragraph+"\n\n", "")
            else:
                section = section.replace(paragraph, "")
    return section


def litllm_predict_related_work_for_data(data:list[dict], model:_BaseLiteratureReviewGenerator, model_name:str, recompute_already_computed:bool=False):
    file_dir = "data/experiment_results/related_work_predictions.json"
    # Read the file first
    if not os.path.isfile(file_dir):
        results = {}
        with open(file_dir, "w") as f:
            json.dump(results, f)
    else: 
        with open(file_dir, "r") as f:
            results = json.load(f)

    for data_point in tqdm(data):
        title = data_point["title"]
        computed = False
        if not recompute_already_computed:
            # Check if it has already been computed with this model
            if title in results:
                if model_name in results[title].keys():
                    if results[title][model_name]["prediction"] != "":
                        computed = True
        # If it has not been computed yet, compute it
        if not computed:
            try:
                if isinstance(model, LitLLMLiteratureReviewGenerator):
                    prediction = model.predict(
                        data_point["abstract"],
                        data_point["rw_in_text_ref_abstracts"],
                        citation_ids = data_point["rw_in_text_ref_nums"],
                        related_work_draft = "",
                        citation_order = extract_citation_order(data_point),
                    )
                elif isinstance(model, LexRankLiteratureReviewGenerator):
                    prediction = model.predict(
                        data_point["abstract"],
                        data_point["rw_in_text_ref_abstracts"],
                        citation_ids = data_point["rw_in_text_ref_nums"],
                    )
                if title not in results:
                    results[title] = {}
                if model_name not in results[title]:
                    results[title][model_name] = {}
                results[title][model_name]["prediction"] = prediction
                # Write the prediction
                with open(file_dir, "w") as f:
                    json.dump(results, f)
            except Exception as e:
                if title not in results:
                    results[title] = {}
                if model_name not in results[title]:
                    results[title][model_name] = {}
                results[title][model_name]["prediction"] = ""
                print(f"Error {e} in paper {title}")
                continue


def compute_literature_review_metric(data:dict, metrics:Union[str, list[str]]=["rouge", "bertscore", "bleurt"], recompute_computed:bool=False):
    if isinstance(metrics, str): metrics = [metrics]
    results_dir = "data/experiment_results/related_work_predictions.json"

    temp = []
    for metric in metrics:
        add = True
        if metric == "rouge":
            print("Loading rouge scorer...", end=" ")
            rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
            print("DONE!")
        elif metric == "bertscore":
            print("Loading BertScore scorer...", end=" ")
            bertscore = load("bertscore")
            print("DONE!")
        elif metric == "bleurt":
            print("Loading Bleurt scorer...", end=" ")
            bleurt = load("bleurt", module_type="metric")
            print("DONE!")
        else:
            print(f"\"{metric}\" is not a defined metric and will be skipped.")
            add = False
        if add:
            temp.append(metric)
    metrics = temp

    with open(results_dir, "r") as f:
        results:dict = json.load(f)
    for paper_title in tqdm(results.keys()):
        truth = ""
        # Find the paper in the data to get the true related work
        for sample in data:
            if sample["title"] == paper_title:
                truth = sample["related_work"]
                break
        for model in results[paper_title].keys():
            prediction = results[paper_title][model]["prediction"]
            if prediction != "":
                if "metrics" not in results[paper_title][model]:
                    results[paper_title][model]["metrics"] = {}
                for metric in metrics:
                    # Check if the metric was already computed
                    computed = False
                    if not recompute_computed:
                        if metric in results[paper_title][model]["metrics"].keys(): 
                            if results[paper_title][model]["metrics"][metric] != None:
                                computed = True
                    # If not yet computed, compute it
                    if not computed:
                        try:    
                            # For Rouge, compute rouge1, rouge2 and rougeL (precision, recall and fmeasure)
                            # Store as dictionary of the format:
                            # {
                            #   "rouge1": {
                            #      "precision": ...
                            #      "recall": ...
                            #      "fmeasure": ...
                            #   }, ...
                            # }
                            if metric == "rouge":
                                score = rouge.score(truth, prediction)
                                value = {}
                                for key in score:
                                    value[key] = {
                                        "precision": score[key].precision,
                                        "recall": score[key].recall,
                                        "fmeasure": score[key].fmeasure
                                    }
                            # For BertScore, compute the precision, recall and f1
                            # Store as dictionary of the format:
                            # {
                            #   "precision": ...
                            #   "recall": ...
                            #   "f1": ...
                            # }
                            elif metric == "bertscore":
                                score = bertscore.compute(predictions=[prediction], references=[truth], lang="en")
                                value = {
                                    "precision": score["precision"][0],
                                    "recall": score["recall"][0],
                                    "f1": score["f1"][0]
                                }
                            # For Bleurt, compute the score
                            # Store as dictionary of the format:
                            # {
                            #   "score": ...
                            # }
                            elif metric == "bleurt":
                                score = bleurt.compute(predictions=[prediction], references=[truth])
                                value = {}
                                value["score"] = score["scores"][0]
                            # Write the prediction
                            with open(results_dir, "w") as f:
                                json.dump(results, f)
                        except Exception as e:
                            print(e)
                            value = None # If there was an error, set the value to None
                        results[paper_title][model]["metrics"][metric] = value


def compute_average_metrics():

    results_dir = "data/experiment_results/related_work_predictions.json"

    with open(results_dir, "r") as f:
        results:dict = json.load(f)

    summarized_results = {}

    for title in results:
        for model in results[title]:
            if "metrics" in results[title][model]:
                for metric in results[title][model]["metrics"]:
                    metric_value = results[title][model]["metrics"][metric]
                    if model not in summarized_results:
                        summarized_results[model] = {}
                    if metric not in summarized_results[model]:
                        if metric == "rouge":
                            summarized_results[model][metric] = {}
                            for rouge_type in metric_value:
                                summarized_results[model][metric][rouge_type] = {
                                    "precision": [metric_value[rouge_type]["precision"]],
                                    "recall": [metric_value[rouge_type]["recall"]],
                                    "fmeasure": [metric_value[rouge_type]["fmeasure"]]
                                }
                        elif metric == "bertscore":
                            summarized_results[model][metric] = {
                                "precision": [metric_value["precision"]],
                                "recall": [metric_value["recall"]],
                                "f1": [metric_value["f1"]]
                            }
                    else:
                        if metric == "rouge":
                            for rouge_type in metric_value:
                                summarized_results[model][metric][rouge_type]["precision"].append(metric_value[rouge_type]["precision"])
                                summarized_results[model][metric][rouge_type]["recall"].append(metric_value[rouge_type]["recall"])
                                summarized_results[model][metric][rouge_type]["fmeasure"].append(metric_value[rouge_type]["fmeasure"])
                        elif metric == "bertscore":
                            summarized_results[model][metric]["precision"].append(metric_value["precision"])
                            summarized_results[model][metric]["recall"].append(metric_value["recall"])
                            summarized_results[model][metric]["f1"].append(metric_value["f1"])
    
    for model in summarized_results:
        print(f"Model: {model}")
        print("-"*len(f"Model: {model}"))
        if "rouge" in summarized_results[model]:
            print("Rouge results:")
            for rouge_type in summarized_results[model]["rouge"]:
                print(f"\t{rouge_type}:")
                print(f'\t\tAverage precision: {np.mean(summarized_results[model]["rouge"][rouge_type]["precision"])}')
                print(f'\t\tAverage recall: {np.mean(summarized_results[model]["rouge"][rouge_type]["recall"])}')
                print(f'\t\tAverage fmeasure: {np.mean(summarized_results[model]["rouge"][rouge_type]["fmeasure"])}')
        if "bertscore" in summarized_results[model]:
            print("BertScore results:")
            print(f'\tAverage precision {np.mean(summarized_results[model]["bertscore"]["precision"])}')
            print(f'\tAverage recall {np.mean(summarized_results[model]["bertscore"]["recall"])}')
            print(f'\tAverage f1 {np.mean(summarized_results[model]["bertscore"]["f1"])}')
        print()

#create_related_work_dataset()

"""
with open(os.path.join(DATA_DIR, "reference_data\\rw_dataset.json"), "r") as f:
    data = json.load(f)

print(f"The dataset has {len(data)} samples")
print("---------------------")
print(f"Example: {data[0]['title']}")
print("---------------------")
print(data[0]["abstract"])
print("---------------------")
print(data[0]["related_work"])
print("---------------------")
print(extract_citation_order(data[0]))
"""

if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")

    with open(os.path.join(DATA_DIR, "reference_data\\rw_dataset.json"), "r") as f:
        data = json.load(f)

    def deepseek_r1_postprocess(out:str):
        thoughts = re.findall(r"<think>[\S\s]*</think>", out)
        for thought in thoughts:
            out = out.replace(thought, "")
        #print(thoughts)
        return out

    deepseek_inference_model = HFClientInferenceModel(
        provider = "novita",
        api_key = HF_TOKEN
    )
    deepseek_inference_model.set_default_call_kwargs(model="deepseek-ai/DeepSeek-R1")
    deepseek_inference_model.postprocess_output = deepseek_r1_postprocess

    models = [
        {
            "model": LitLLMLiteratureReviewGenerator(deepseek_inference_model, method="vanilla"),
            "model_name": "deepseek-r1:685B (vanilla)"
        },

        #{
        #    "model": LitLLMLiteratureReviewGenerator(deepseek_inference_model, method="sentence"),
        #    "model_name": "deepseek-r1:685B (sentence)"
        #},

        {
            "model": LexRankLiteratureReviewGenerator(),
            "model_name": "lexrank"
        }
    ]

    hf_models = [
        {
            "hf_name": "deepseek-ai/DeepSeek-R1",
            "model_name": "deepseek-r1:685B",
            "provider": "novita",
            "postprocess": deepseek_r1_postprocess
        }
    ]

    #compute_literature_review_metric(data, metrics=["bertscore"])

    compute_average_metrics()

    #for model in models:
    #    litllm_predict_related_work_for_data(
    #        data,
    #        model["model"],
    #        model["model_name"]
    #    )
