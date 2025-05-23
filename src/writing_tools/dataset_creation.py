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


def create_reference_dataframe(references_info_df:pd.DataFrame=pd.read_csv(os.path.join(DATA_DIR, "reference_data/refs.csv")),
                               save_dir:str=os.path.join(os.path.join(DATA_DIR, "reference_data/rw_data")), 
                               save_name:str="ref_data.csv"):
    references_df = pd.DataFrame(columns=["oaid", "title", "abstract"]) # Construct the new DataFrame
    # Loop through all references
    for i in tqdm(references_info_df.index):
        oaid = references_info_df.loc[i, "oaid"] # Fetch the oaid
        abstract = references_info_df.loc[i, "abstract"] # Fetch the abstract
        title = references_info_df.loc[i, "title"] # Fetch the title
        # If the abstract is available, store the information into the new DataFrame
        if isinstance(abstract, str) and len(abstract) > 0 and isinstance(title, str) and len(title) > 0 and isinstance(oaid, str) and len(oaid) > 0:
            # Make sure not to store duplicate OAIDs/titles
            if (oaid not in references_df["oaid"].tolist()) and (title not in references_df["title"].tolist()):
                references_df.loc[len(references_df)] = [
                    oaid, 
                    title, 
                    abstract
                ]
    references_df.to_csv(os.path.join(save_dir, save_name), index=False) # Save the new DataFrame
    return references_df


def get_paper_markdowns(dir:str=os.path.join(DATA_DIR, "challenge10_batch_1\\CVPR_2024\\Conversions\\opencvf-data\\md")):
    paper_markdowns = []
    for paper in tqdm(os.listdir(dir), desc="Collecting MD paper texts..."):
        with open(os.path.join(dir, paper), "r", encoding="utf-8") as p:
            paper_text = p.read() # Fetch the markdown paper text
        paper_markdowns.append((paper.replace(".md", ""), paper_text))
    return paper_markdowns


def create_paper_dataframe(papers_info_df:pd.DataFrame=pd.read_csv(os.path.join(DATA_DIR, "reference_data/orig.csv")),
                           paper_markdowns:list[str]=get_paper_markdowns(), 
                           save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), 
                           save_name:str="paper_df.csv"):
    paper_df = pd.DataFrame(columns=["oaid", "title", "related_work", "abstract", "references"]) # new DataFrame of paper information
    # Loop through all the papers
    for paper_name, paper_text in tqdm(paper_markdowns, desc="Creating paper/reference data"):
        paper_info = papers_info_df[papers_info_df["fname"] == paper_name+".txt"] # Fetch the info about the current paper
        # If paper is not found, skip it
        try:
            paper_title = paper_info["title"].values[0] # Fetch the title of the paper
            paper_oaid = paper_info["oaid"].values[0] # Fetch the paper oaid
        except:
            continue
        # Find all the different sections
        sections = paper_text.split("##") 
        sections = ["##" + section for section in sections]
        # Find the relevant sections
        abstract_section = None
        related_work_section = None
        references_section = None
        for section in sections:
            if section.startswith("## Abstract"):
                abstract_section = clean_section(section.replace("## Abstract", ""))
            if section.startswith("## References"):
                references_section = clean_section(section.replace("## References", ""))
            rw_titles = re.findall(r"## [0-9]+\. Related Work", section)
            if len(rw_titles) > 0:
                related_work_section = clean_section(section.replace(rw_titles[0], ""))
        # If all the sections were found, proceed
        if isinstance(abstract_section, str) and isinstance(related_work_section, str) and isinstance(references_section, str) and isinstance(paper_oaid, str) and\
            len(abstract_section) > 0 and len(related_work_section) > 1 and len(references_section) > 0 and len(paper_oaid) > 0:
            paper_df.loc[len(paper_df)] = [paper_oaid, paper_title, related_work_section, abstract_section, references_section]
    paper_df.to_csv(os.path.join(save_dir, save_name), index=False)
    return paper_df


def create_paper_reference_connection_dataframe(references_df:pd.DataFrame=pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/ref_df.csv")), 
                                      paper_df:pd.DataFrame=pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_df.csv")),
                                      save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"),
                                      save_name:str="paper_rw_ref_df.csv"):
    
    paper_rw_ref_df = pd.DataFrame(columns=["paper_oaid", "ref_oaid", "ref_num", "ref_position", "ref_sentence"]) # DataFrame of paper reference connections
    # Loop through all the papers
    for paper_idx in tqdm(paper_df.index, desc="Creating paper/reference data"):
        paper_oaid = paper_df.loc[paper_idx, "oaid"] # Fetch the oaid of the paper
        related_work = paper_df.loc[paper_idx, "related_work"] # Fetch the Related Work section
        references = paper_df.loc[paper_idx, "references"] # Fetch the References section
        # Extract the in text reference numbers from the Related Work section (in order) and parse them
        try:
            rw_in_text_ref_nums_order_strings = re.findall(r"\[[0-9, -]+\]", related_work) # Find the respective strings
        except:
            continue
        rw_in_text_ref_nums_order = []
        for i, rw_in_text_ref_num_string in enumerate(rw_in_text_ref_nums_order_strings):
            # Look for "5-8" reference types
            range_refs = re.findall(r"[0-9]+-[0-9]+", rw_in_text_ref_num_string)
            #if len(range_refs) > 0:
                #print(rw_in_text_ref_num_string)
            for range_ref in range_refs:
                minus_pos = range_ref.find("-")
                a = int(range_ref[:minus_pos])
                b = int(range_ref[minus_pos+1:])
                rw_in_text_ref_num_string = rw_in_text_ref_num_string.replace(range_ref, str([i for i in range(a, b+1)]).replace("[", "").replace("]", ""))
                #print(rw_in_text_ref_num_string)
                #print()
            try:
                rw_in_text_ref_nums_order.append([int(ref) for ref in ast.literal_eval(rw_in_text_ref_num_string)]) # Parse the string to lists of integers
            except:
                print(rw_in_text_ref_num_string)
                return
        # Extract the reference numbers and row strings from the References section
        reference_rows = references.split("\n")
        all_refs = {}
        for reference_row in reference_rows:
            # If the row has the reference number, take it, otherwise keep looping
            row_ref_nums = re.findall(r"\[([0-9]+)\]", reference_row)
            if len(row_ref_nums) > 0:
                all_refs[int(row_ref_nums[0])] = reference_row
            else:
                continue
        # Extract all the sentences in the Related Work section
        rw_sentences = related_work.split(".")
        rw_sentences = [sentence.strip("\n ")+"." for sentence in rw_sentences]
        # Loop through the in-text references (in order)
        for ref_position, rw_in_text_ref_nums in enumerate(rw_in_text_ref_nums_order):
            # Find the sentence it is in
            in_text_ref_str = rw_in_text_ref_nums_order_strings[ref_position]
            in_text_ref_str = "\\" + in_text_ref_str[:-1] + "\\]"
            try:
                ref_sentence = re.findall(r"\.[\n ]*([^\.]*" + in_text_ref_str + r"[^\.]*\.)", "."+related_work)[0]
            except:
                continue
            #ref_sentence = None
            #for sentence in rw_sentences:
            #    if str(rw_in_text_ref_nums) in sentence:
            #        ref_sentence = sentence
            #        break
            # Loop through all the reference numbers
            for ref_num in rw_in_text_ref_nums:
                try:
                    if ref_num in all_refs:
                        q = references_df.apply(lambda row: isinstance(row["title"], str) and (row["title"].lower() == all_refs[ref_num].split(".")[1].strip().lower()), axis=1)
                        #print(references_df["title"])
                        if q.any():
                            ref_info = references_df[q].copy()
                            # Sort by the length of the title, and take the longest one
                            ref_info["title_length"] = [len(title) for title in ref_info["title"]]
                            ref_info.sort_values(by="title_length", ascending=False, inplace=True)
                            #print(all_refs[ref_num].split(".")[1].strip())
                            #print(ref_info["title"])
                            #print()
                            ref_oaid = ref_info["oaid"].values[0]
                            paper_rw_ref_df.loc[len(paper_rw_ref_df)] = [paper_oaid, ref_oaid, ref_num, ref_position, ref_sentence]
                except:
                    continue
                # Find the reference by title and save it if found
                #if ref_num in all_refs:
                #    for j in references_df.index:
                #        ref_title = references_df.loc[j, "title"]
                #        if isinstance(ref_title, str) and ref_title in all_refs[ref_num]:
                #            ref_oaid = references_df.loc[j, "oaid"]
                #            paper_rw_ref_df.loc[len(paper_rw_ref_df)] = [paper_oaid, ref_oaid, ref_num, ref_position, ref_sentence]
                #            break
            #print()
            #print(paper_rw_ref_df.head(10))
            #print(ref_sentence)

    paper_rw_ref_df.to_csv(os.path.join(save_dir, save_name), index=False)
    return paper_rw_ref_df


def create_luis_sentence_dataset(paper_rw_ref_df:pd.DataFrame=pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_rw_ref_df.csv")),
                                 save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"),
                                 save_name:str="luis.json"):
    unique_sentences = paper_rw_ref_df["ref_sentence"].unique()
    data = []
    for sentence in tqdm(unique_sentences):
        ref_oaids = list(set(paper_rw_ref_df.loc[paper_rw_ref_df["ref_sentence"] == sentence, "ref_oaid"].tolist()))
        paper_oaid = paper_rw_ref_df.loc[paper_rw_ref_df["ref_sentence"] == sentence, "paper_oaid"].values[0]
        data.append({
            "sentence": sentence,
            "ref_oaids": ref_oaids,
            "paper_oaid": paper_oaid
        })
    with open(os.path.join(save_dir, save_name), "w") as f:
        json.dump(data, f)
    return data

            
            


def _create_related_work_dataset():
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
    #   "rw_in_text_ref_nums": ["integer"],
    #   "rw_in_text_ref_abstracts": ["string"],
    #   "rw_draft": "string" # Related work up to the second citation,
    #   "rw_next_in_text_ref_nums": ["integer"],
    #   "rw_next_sentence": "string"
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
            for section in sections:
                if related_work_title in section:
                    related_work_section = section
                elif references_title in section:
                    references_section = section
                elif abstract_title in section:
                    abstract_section = section
            # Skip if either of the sections is empty
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
            # For sentence by sentence generation, find the second sentence that has in text citations that are available
            related_work_sentences = related_work_section.split(".")
            count_found_citation_sentences = 0
            related_work_draft = ""
            for sentence in related_work_sentences:
                sentence_in_text_citations = re.findall(r"\[[0-9, ]+\]", sentence)
                if len(sentence_in_text_citations) > 0:
                    # Check if the citations are available
                    all_sentence_in_text_citations = []
                    for s in sentence_in_text_citations:
                        all_sentence_in_text_citations += [int(i) for i in ast.literal_eval(s)]
                    for ref in all_sentence_in_text_citations:
                        if ref in related_work_ref_nums:
                            count_found_citation_sentences += 1
                            if count_found_citation_sentences == 2:
                                dict_new["rw_next_in_text_ref_nums"] = []
                                # Add all the references that have an available abstract
                                for ref2 in all_sentence_in_text_citations:
                                    if ref2 in related_work_ref_nums:
                                        dict_new["rw_next_in_text_ref_nums"].append(ref2)
                                # Add the sentence
                                dict_new["rw_next_sentence"] = sentence.strip("\n ") + "."
                            else:
                                related_work_draft += sentence + "."
                            break
                if count_found_citation_sentences == 2:
                    dict_new["rw_draft"] = related_work_draft
                    break
            if count_found_citation_sentences < 2:
                dict_new["rw_next_in_text_ref_nums"] = []
                dict_new["rw_next_sentence"] = ""
                dict_new["rw_draft"] = ""
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
    #section = section.replace(paragraphs[0]+"\n\n", "")
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith("Figure ") or paragraph.startswith("Table ") or (paragraph.startswith("|") and paragraph.endswith("|")):
            if i < len(paragraphs)-1:
                section = section.replace(paragraph+"\n\n", "")
            else:
                section = section.replace(paragraph, "")
    return section


def predict_related_work_for_data(data:list[dict], model:_BaseLiteratureReviewGenerator, model_name:str, recompute_already_computed:bool=False):
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
                    if model.method == "vanilla":
                        prediction = model.predict(
                            data_point["abstract"],
                            data_point["rw_in_text_ref_abstracts"],
                            citation_ids = data_point["rw_in_text_ref_nums"],
                            related_work_draft = "",
                            citation_order = extract_citation_order(data_point),
                        )
                    elif model.method == "sentence":
                        if data_point["rw_draft"] != "":
                            ref_abstracts = []
                            for i, ref in enumerate(data_point["rw_in_text_ref_nums"]):
                                if ref in data_point["rw_next_in_text_ref_nums"]:
                                    ref_abstracts.append(data_point["rw_in_text_ref_abstracts"][i])
                            prediction = model.predict_next(
                                data_point["abstract"],
                                ref_abstracts,
                                citation_ids = data_point["rw_next_in_text_ref_nums"],
                                related_work_draft = data_point["rw_draft"]
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


def compute_literature_review_metric(data:dict, metrics:Union[str, list[str]]=["rouge", "bertscore", "bleurt", "p_reference"], recompute_computed:bool=False):
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
        elif metric == "p_reference":
            pass
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
        paper = {}
        # Find the paper in the data to get the true related work
        for sample in data:
            if sample["title"] == paper_title:
                paper = sample
                break
        for model in results[paper_title].keys():
            truth = paper["related_work"] if "(sentence)" not in model else paper["rw_next_sentence"]
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
                            # The percentage of references that were included in the prediction
                            elif metric == "p_reference":
                                # Find the references included
                                ref_lists = re.findall(r"\[[0-9, ]+\]", prediction)
                                included_refs = set()
                                for ref_list in ref_lists:
                                    ref_list = ast.literal_eval(ref_list)
                                    for ref in ref_list:
                                        included_refs.add(int(ref))
                                all_rw_refs = set(paper["rw_in_text_ref_nums"])
                                true_refs = set(paper["rw_in_text_ref_nums"]) if "(sentence)" not in model else set(paper["rw_next_in_text_ref_nums"])
                                correct_refs = []
                                incorrect_in_rw_refs = []
                                incorrect_not_rw_refs = []
                                for ref in included_refs:
                                    if ref in true_refs:
                                        correct_refs.append(ref)
                                    elif ref in all_rw_refs:
                                        incorrect_in_rw_refs.append(ref)
                                    else:
                                        incorrect_not_rw_refs.append(ref)
                                value = {
                                    "total": len(included_refs)/len(true_refs) if len(true_refs) > 0 else 0,
                                    "correct": len(correct_refs)/len(true_refs) if len(true_refs) > 0 else 0,
                                    "incorrect_in_rw": len(incorrect_in_rw_refs)/len(true_refs) if len(true_refs) > 0 else 0,
                                    "incorrect_not_rw": len(incorrect_not_rw_refs)/len(true_refs) if len(true_refs) > 0 else 0 
                                }
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
                        elif metric == "p_reference":
                            summarized_results[model][metric] = {
                                "total": [metric_value["total"]],
                                "correct": [metric_value["correct"]],
                                "incorrect_in_rw": [metric_value["incorrect_in_rw"]],
                                "incorrect_not_rw": [metric_value["incorrect_not_rw"]]
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
                        elif metric == "p_reference":
                            if metric_value is not None and "total" in metric_value:
                                summarized_results[model][metric]["total"].append(metric_value["total"])
                                summarized_results[model][metric]["correct"].append(metric_value["correct"])
                                summarized_results[model][metric]["incorrect_in_rw"].append(metric_value["incorrect_in_rw"])
                                summarized_results[model][metric]["incorrect_not_rw"].append(metric_value["incorrect_not_rw"])
                        
    
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
        if "p_reference" in summarized_results[model]:
            print("Number of references included")
            print(f'\tTotal predicted references / true references {np.mean(summarized_results[model]["p_reference"]["total"])}')
            print(f'\tTrue predicted references / true references {np.mean(summarized_results[model]["p_reference"]["correct"])}')
            print(f'\tIncorrect, but in Related Work predicted references / true references {np.mean(summarized_results[model]["p_reference"]["incorrect_in_rw"])}')
            print(f'\tIncorrect, halucinated predicted references / true references {np.mean(summarized_results[model]["p_reference"]["incorrect_not_rw"])}')
        print()


def prep_paper_for_prediction(paper_oaid:str, 
                              paper_df:pd.DataFrame,
                              paper_rw_ref_df:pd.DataFrame,
                              ref_df:pd.DataFrame):
    # Find the info for the entire rw prediction
    paper_abstract = paper_df.loc[paper_df["oaid"] == paper_oaid, "abstract"].values[0]
    related_work = paper_df.loc[paper_df["oaid"] == paper_oaid, "related_work"].values[0]
    paper_references_df = paper_rw_ref_df.loc[paper_rw_ref_df["paper_oaid"] == paper_oaid]
    # If there are no references, return None
    if len(paper_references_df) == 0:
        return None
    # To make sure that the references aren't duplicated, group them by reference number and oaid
    ref_info_grouped = paper_references_df.groupby(["ref_num", "ref_oaid"]).count().reset_index()
    rw_in_text_ref_nums = ref_info_grouped["ref_num"].tolist()
    rw_in_text_ref_oaids = ref_info_grouped["ref_oaid"].tolist()
    rw_in_text_ref_abstracts = []
    for ref_oaid in rw_in_text_ref_oaids:
        rw_in_text_ref_abstracts.append(ref_df.loc[ref_df["oaid"] == ref_oaid, "abstract"].values[0])
    # Find the info for the next sentence prediction
    # Check how many different sentences have citations in the paper
    rw_draft = None
    next_sentence = None
    rw_next_in_text_ref_nums = None
    if len(paper_references_df["ref_sentence"].unique()) > 1:
        # Find the second reference position
        positions = sorted(paper_references_df["ref_position"].unique().tolist(), reverse=False)
        second_pos = positions[1]
        next_sentence = paper_references_df.loc[paper_references_df["ref_position"] == second_pos, "ref_sentence"].values[0]
        rw_draft = related_work[:related_work.find(next_sentence)]
        rw_next_in_text_ref_nums = paper_references_df.loc[paper_references_df["ref_position"] == second_pos, "ref_num"].unique().tolist()
    return {
        "paper_oaid": paper_oaid,
        "paper_abstract": paper_abstract,
        "related_work": related_work,
        "rw_in_text_ref_nums": rw_in_text_ref_nums,
        "rw_in_text_ref_abstracts": rw_in_text_ref_abstracts,
        "rw_next_in_text_ref_nums": rw_next_in_text_ref_nums,
        "rw_draft": rw_draft,
        "rw_next_sentence": next_sentence
    }


def create_related_work_dataset(paper_df:pd.DataFrame, paper_rw_ref_df:pd.DataFrame, ref_df:pd.DataFrame):
    data = []
    for paper_oaid in tqdm(paper_df["oaid"].tolist()):
        #print(paper_oaid)
        if paper_oaid is not None:
            paper_dict = prep_paper_for_prediction(paper_oaid, paper_df, paper_rw_ref_df, ref_df)
            if paper_dict is not None:
                data.append(paper_dict)
    return data


def predict_entire_rw(data:dict, model:_BaseLiteratureReviewGenerator, model_save_name:str,
                      save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), 
                      save_name:str="prediction_df.csv", recompute_computed:bool=False):
    # Load already existing predictions (if they exist)
    try:
        prediction_df = pd.read_csv(os.path.join(save_dir, save_name))
    except:
        prediction_df = pd.DataFrame(columns=["model", "type", "paper_oaid", "prediction"])
    for paper_dict in tqdm(data, desc=f"Predicting the entire Related Work sections with {model_save_name}"):
        compute = True
        if not recompute_computed:
            compute = not ((prediction_df["model"] == model_save_name) & (prediction_df["type"] == "entire") & (prediction_df["paper_oaid"] == paper_dict["paper_oaid"])).any()
        if compute:
            try:
                prediction = model.predict_entire(
                    paper_dict["paper_abstract"],
                    paper_dict["rw_in_text_ref_abstracts"],
                    paper_dict["rw_in_text_ref_nums"]
                )
                # Add the None values to compensate for other columns
                prediction_df.loc[len(prediction_df)] = [model_save_name, "entire", paper_dict["paper_oaid"], prediction] + [None]*(len(prediction_df.columns)-4) 
                prediction_df.to_csv(os.path.join(save_dir, save_name), index=False)
            except Exception as e:
                print(e)
                continue


def predict_next_sentence_rw(data:dict, model:_BaseLiteratureReviewGenerator, model_save_name:str,
                      save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), 
                      save_name:str="prediction_df.csv", recompute_computed:bool=False):
    # Load already existing predictions (if they exist)
    try:
        prediction_df = pd.read_csv(os.path.join(save_dir, save_name))
    except:
        prediction_df = pd.DataFrame(columns=["model", "type", "paper_oaid", "prediction"])
    for paper_dict in tqdm(data, desc=f"Predicting the next sentence in the Related Work sections with {model_save_name}"):
        compute = True
        if not recompute_computed:
            compute = not ((prediction_df["model"] == model_save_name) & (prediction_df["type"] == "sentence") & (prediction_df["paper_oaid"] == paper_dict["paper_oaid"])).any()
        if compute:
            try:
                if paper_dict["rw_next_in_text_ref_nums"] is not None:
                    rw_next_in_text_ref_abstracts = []
                    for ref_num in paper_dict["rw_next_in_text_ref_nums"]:
                        pos = paper_dict["rw_in_text_ref_nums"].index(ref_num)
                        rw_next_in_text_ref_abstracts.append(paper_dict["rw_in_text_ref_abstracts"][pos])
                    prediction = model.predict_next(
                        paper_dict["paper_abstract"],
                        rw_next_in_text_ref_abstracts,
                        paper_dict["rw_next_in_text_ref_nums"],
                        paper_dict["rw_draft"]
                    )
                    # Add the None values to compensate for other columns
                    prediction_df.loc[len(prediction_df)] = [model_save_name, "sentence", paper_dict["paper_oaid"], prediction] + [None]*(len(prediction_df.columns)-4) 
                    prediction_df.to_csv(os.path.join(save_dir, save_name), index=False)
            except Exception as e:
                print(e)
                continue


def compute_prediction_rouge(data:dict, prediction_df:pd.DataFrame, save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), save_name:str="prediction_rouge_df.csv", recompute_computed:bool=False):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    prediction_df = prediction_df.copy()

    # Make sure to add the Rouge columns
    for rouge_type in scorer.rouge_types:
        for metric_type in ["precision", "recall", "fmeasure"]:
            c = f"rouge.{rouge_type}.{metric_type}"
            if c not in prediction_df.columns:
                prediction_df[c] = None

    # Loop through the data
    for paper in tqdm(data, "Predicting Rouge for papers..."):
        paper_oaid = paper["paper_oaid"]
        # Find the paper in the predictions
        paper_prediction_rows = prediction_df.loc[prediction_df["paper_oaid"] == paper_oaid]
        for i in paper_prediction_rows.index:
            # Check if prediction exists, and if not, skip this
            if not isinstance(paper_prediction_rows.loc[i, "prediction"], str):
                continue
            # Check if already computed
            compute = True
            if not recompute_computed:
                metric_cols = []
                for rouge_type in scorer.rouge_types:
                    for metric_type in ["precision", "recall", "fmeasure"]:
                        metric_cols.append(f"rouge.{rouge_type}.{metric_type}")
                if not paper_prediction_rows.loc[i, metric_cols].isna().any():
                    compute = False
            # Compute the metric
            if compute:
                # Find the truth depending on the method
                if paper_prediction_rows.loc[i, "type"] == "sentence":
                    truth = paper["rw_next_sentence"]
                elif paper_prediction_rows.loc[i, "type"] == "entire":
                    truth = paper["related_work"]
                else:
                    continue
                # Compute the Rouge
                score = scorer.score(truth, paper_prediction_rows.loc[i, "prediction"])
                for rouge_type in score.keys():
                    prediction_df.loc[i, f"rouge.{rouge_type}.precision"] = score[rouge_type].precision
                    prediction_df.loc[i, f"rouge.{rouge_type}.recall"] = score[rouge_type].recall
                    prediction_df.loc[i, f"rouge.{rouge_type}.fmeasure"] = score[rouge_type].fmeasure
    prediction_df.to_csv(os.path.join(save_dir, save_name), index=False)
    return prediction_df


def compute_prediction_bertscore(data:dict, prediction_df:pd.DataFrame, save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), save_name:str="prediction_bertscore_df.csv", recompute_computed:bool=False):
    scorer = load("bertscore")

    prediction_df = prediction_df.copy()

    # Make sure to add the Bertscore columns
    for metric_type in ["precision", "recall", "f1"]:
        c = f"bertscore.{metric_type}"
        if c not in prediction_df.columns:
            prediction_df[c] = None

    # Loop through the data
    for paper in tqdm(data, "Predicting BertScore for papers..."):
        paper_oaid = paper["paper_oaid"]
        # Find the paper in the predictions
        paper_prediction_rows = prediction_df.loc[prediction_df["paper_oaid"] == paper_oaid]
        for i in paper_prediction_rows.index:
            # Check if prediction exists, and if not, skip this
            if not isinstance(paper_prediction_rows.loc[i, "prediction"], str):
                continue
            # Check if already computed
            compute = True
            if not recompute_computed:
                metric_cols = []
                for metric_type in ["precision", "recall", "f1"]:
                    metric_cols.append(f"bertscore.{metric_type}")
                if not paper_prediction_rows.loc[i, metric_cols].isna().any():
                    compute = False
            # Compute the metric
            if compute:
                # Find the truth depending on the method
                if paper_prediction_rows.loc[i, "type"] == "sentence":
                    truth = paper["rw_next_sentence"]
                elif paper_prediction_rows.loc[i, "type"] == "entire":
                    truth = paper["related_work"]
                else:
                    continue
                # Compute the BertScore
                score = scorer.compute(predictions=[paper_prediction_rows.loc[i, "prediction"]], references=[truth], lang="en")
                for metric_type in score.keys():
                    prediction_df.loc[i, f"bertscore.{metric_type}"] = score[metric_type][0]
    prediction_df.to_csv(os.path.join(save_dir, save_name), index=False)
    return prediction_df


def compute_prediction_p_reference(data:dict, prediction_df:pd.DataFrame, save_dir:str=os.path.join(DATA_DIR, "reference_data/rw_data"), save_name:str="prediction_p_reference_df.csv", recompute_computed:bool=False):
    prediction_df = prediction_df.copy()

    """
    p_correct: % of the true references that are included (best case 1.0)
    p_incorrect: % of made up references that are included (best case 0.0)
    """

    # Make sure to add the Bertscore columns
    metric_types = ["p_correct", "p_incorrect"]
    for metric_type in metric_types:
        c = f"p_refernece.{metric_type}"
        if c not in prediction_df.columns:
            prediction_df[c] = None

    # Loop through the data
    for paper in tqdm(data, "Predicting p reference for papers..."):
        paper_oaid = paper["paper_oaid"]
        # Find the paper in the predictions
        paper_prediction_rows = prediction_df.loc[prediction_df["paper_oaid"] == paper_oaid]
        for i in paper_prediction_rows.index:
            # Check if prediction exists, and if not, skip this
            if not isinstance(paper_prediction_rows.loc[i, "prediction"], str):
                continue
            # Check if already computed
            compute = True
            if not recompute_computed:
                metric_cols = []
                for metric_type in metric_types:
                    metric_cols.append(f"bertscore.{metric_type}")
                if not paper_prediction_rows.loc[i, metric_cols].isna().any():
                    compute = False
            # Compute the metric
            if compute:
                # Extract the predicted reference numbers
                prediction = extract_reference_numbers(paper_prediction_rows.loc[i, "prediction"])
                # Find the truth depending on the method
                if paper_prediction_rows.loc[i, "type"] == "sentence":
                    truth = paper["rw_next_in_text_ref_nums"]
                    truth_previous_temp = extract_reference_numbers(paper["rw_draft"])
                    truth_previous = []
                    for ref in truth_previous_temp:
                        if ref not in truth:
                            truth_previous.append(ref)
                elif paper_prediction_rows.loc[i, "type"] == "entire":
                    truth = paper["rw_in_text_ref_nums"]
                    truth_previous = []
                else:
                    continue
                # Compute the p refernece
                prediction_correct = []
                prediction_incorrect = []
                prediction_previous = []
                for ref in prediction:
                    if ref in truth:
                        prediction_correct.append(ref)
                    else:
                        prediction_incorrect.append(ref)
                        if ref in truth_previous:
                            prediction_previous.append(ref)
                prediction_df.loc[i, "p_reference.p_correct_predictions"] = len(prediction_correct)/len(prediction) 
                prediction_df.loc[i, "p_reference.p_predicted_truths"] = len(prediction_correct)/len(truth)
                prediction_df.loc[i, "p_reference.p_previous_predictions_incorrect_predictions"] = len(prediction_previous)/len(prediction_incorrect)
    prediction_df.to_csv(os.path.join(save_dir, save_name), index=False)
    return prediction_df

#create_related_work_dataset()

def extract_reference_numbers(text:str):
    ref_nums = set()
    ref_list_strings = re.findall(r"\[[0-9, -]+\]", text)
    for ref_list_string in ref_list_strings:
        # Check for range references
        range_refs = re.findall(r"[0-9]+-[0-9]+", ref_list_string)
        ref_list_string_cleaned = ref_list_string
        for range_ref in range_refs:
            a = int(range_ref[:range_ref.find("-")])
            b = int(range_ref[range_ref.find("-")+1:])
            ref_list_string_cleaned.replace(range_ref, ", ".join([i for i in range(a, b+1)]))
        # Try evaluating the list
        try:
            ref_list = [int(i) for i in ast.literal_eval(ref_list_string_cleaned)]
            for ref in ref_list:
                ref_nums.add(ref)
        except:
            continue
    return list(ref_nums)


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
    HF_TOKEN = os.environ.get("HF_TOKEN") # Fetch the Hugging Face token

    # Creating the datasets
    #ref_df = create_reference_dataframe()
    #paper_df = create_paper_dataframe()
    #paper_rw_ref_df = create_paper_reference_connection_dataframe(references_df=ref_df, paper_df=paper_df)
    prediction_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/prediction_df.csv"))
    print(prediction_df[prediction_df["prediction"].isna()])

    # Load the datasets
    ref_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/ref_df.csv"))
    paper_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_df.csv"))
    paper_rw_ref_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_rw_ref_df.csv"))

    # Create the dataset for predicting the entire Related Work section
    data = create_related_work_dataset(paper_df, paper_rw_ref_df, ref_df)

    # Create the models that will be used
    llama_31_8B_instruct_inference_model = HFClientInferenceModel(
        provider = "nebius",
        api_key = HF_TOKEN
    )
    llama_31_8B_instruct_inference_model.set_default_call_kwargs(model="meta-llama/Llama-3.1-8B-Instruct")

    llama_31_70B_instruct_inference_model = HFClientInferenceModel(
        provider = "nebius",
        api_key = HF_TOKEN
    )
    llama_31_70B_instruct_inference_model.set_default_call_kwargs(model="meta-llama/Llama-3.1-70B-Instruct")

    llama_31_405B_instruct_inference_model = HFClientInferenceModel(
        provider = "nebius",
        api_key = HF_TOKEN
    )
    llama_31_405B_instruct_inference_model.set_default_call_kwargs(model="meta-llama/Llama-3.1-405B-Instruct")

    models = [
        #{
        #    "model": LitLLMLiteratureReviewGenerator(llama_31_8B_instruct_inference_model),
        #    "model_name": "llama-3.1-instruct:8B"
        #},

        {
            "model": LitLLMLiteratureReviewGenerator(llama_31_70B_instruct_inference_model),
            "model_name": "llama-3.1-instruct:70B"
        },

        #{
        #    "model": LitLLMLiteratureReviewGenerator(llama_31_405B_instruct_inference_model),
        #    "model_name": "llama-3.1-instruct:405B"
        #},

        #{
        #    "model": LexRankLiteratureReviewGenerator(),
        #    "model_name": "lexrank"
        #}
    ]

    # Predict the next sentence in the related work
    #for model in models:
    #    predict_next_sentence_rw(
    #        data,
    #        model["model"],
    #        model["model_name"]
    #    )
    # Predict the entire related work
    #for model in models:
    #    predict_entire_rw(
    #        data,
    #        model["model"],
    #        model["model_name"]
    #    )
    
    compute_prediction_bertscore(data, prediction_df)

    #create_related_work_dataset()
    #pprint.pprint(data[0])

    #compute_literature_review_metric(data, metrics=["p_reference"], recompute_computed=True)

    #paper_df = create_paper_dataframe()
    #print(len(paper_df))
    #print(paper_df.head())
    #paper_rw_ref_df = create_paper_reference_connection_dataframe()
    #print(len(paper_rw_ref_df))
    #print(paper_rw_ref_df.head())
    #ref_df = create_reference_dataframe()
    #counts = ref_df["title"].value_counts(sort=True, ascending=False)
    #print(len(ref_df))
    #print(counts[counts > 1])
    #data = create_luis_sentence_dataset()
    #print(len(data))
    #pprint.pprint(data[0])

    #with open(os.path.join(DATA_DIR, "reference_data\\rw_dataset.json"), "r") as f:
    #    data = json.load(f)

    #def deepseek_r1_postprocess(out:str):
    #    thoughts = re.findall(r"<think>[\S\s]*</think>", out)
    #    for thought in thoughts:
    #        out = out.replace(thought, "")
    #    #print(thoughts)
    #    return out

    #deepseek_inference_model = HFClientInferenceModel(
    #    provider = "novita",
    #    api_key = HF_TOKEN
    #)
    #deepseek_inference_model.set_default_call_kwargs(model="deepseek-ai/DeepSeek-R1")
    #deepseek_inference_model.postprocess_output = deepseek_r1_postprocess

    paper_rw_ref_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_rw_ref_df.csv"))
    paper_rw_ref_df_grouped = paper_rw_ref_df.groupby(["paper_oaid", "ref_oaid"]).count().reset_index()
    print(paper_rw_ref_df_grouped.head())
    paper_counts = paper_rw_ref_df_grouped["paper_oaid"].value_counts(sort=True, ascending=True)
    print(paper_counts.head())
    print(len(paper_counts[paper_counts < 5]))

    paper_counts_df = pd.DataFrame(index=paper_counts.index)
    paper_counts_df["count"] = paper_counts.values
    print(paper_counts_df.head())
    paper_df = pd.read_csv(os.path.join(DATA_DIR, "reference_data/rw_data/paper_df.csv"))
    for paper_oaid in paper_counts_df.index:
        rw = paper_df.loc[paper_df["oaid"] == paper_oaid, "related_work"].values[0]
        try:
            rw_in_text_ref_nums_order_strings = re.findall(r"\[[0-9, -]+\]", rw) # Find the respective strings
        except:
            #print(paper_df.loc[i, "title"])
            print(rw)
            continue
        rw_in_text_ref_nums_order = []
        in_text_refs = []
        for i, rw_in_text_ref_num_string in enumerate(rw_in_text_ref_nums_order_strings):
            # Look for range reference types
            range_refs = re.findall(r"[0-9]+-[0-9]+", rw_in_text_ref_num_string)
            for range_ref in range_refs:
                minus_pos = range_ref.find("-")
                a = int(range_ref[:minus_pos])
                b = int(range_ref[minus_pos+1:])
                rw_in_text_ref_num_string = rw_in_text_ref_num_string.replace(range_ref, str([i for i in range(a, b+1)]).replace("[", "").replace("]", ""))
            try:
                in_text_refs += [int(ref) for ref in ast.literal_eval(rw_in_text_ref_num_string)] # Parse the string to lists of integers
            except:
                print(rw_in_text_ref_num_string)
        in_text_refs = list(set(in_text_refs))
        paper_counts_df.loc[paper_oaid, "true_count"] = len(in_text_refs)
    paper_counts_df["p"] = paper_counts_df["count"]/paper_counts_df["true_count"]
    print(paper_counts_df.head())

    #paper_counts_df = paper_counts_df[paper_counts_df["true_count"] > 0]


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].boxplot(paper_counts_df["count"])
    ax[1, 0].hist(paper_counts_df["count"])
    ax[0, 1].boxplot(paper_counts_df["true_count"])
    ax[1, 1].hist(paper_counts_df["true_count"])
    ax[0, 2].boxplot(paper_counts_df["p"])
    ax[1, 2].hist(paper_counts_df["p"])
    plt.show()
    #compute_average_metrics()

    #for model in models:
    #    predict_related_work_for_data(
    #        data,
    #        model["model"],
    #        model["model_name"]
    #    )
