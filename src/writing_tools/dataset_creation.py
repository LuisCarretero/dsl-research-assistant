import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import re
import ast
from pyalex import Works
import pandas as pd
import pprint

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

def add_related_work_refs_to_orig_df():
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
        related_work_titles = re.findall("## ([0-9]. Related Work)", paper_text)
        # Find the references section title
        references_titles = re.findall("## (References)", paper_text)
        # Find the abstract section title
        abstract_titles = re.findall("## (Abstract)", paper_text)
        
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
                    abstract_section = ""
            dict_new = {
                "title": orig_paper["title"].values[0],
                "abstract": abstract_section,
                "related_work": related_work_section,
                "rw_in_text_ref_nums": [],
                "rw_in_text_ref_abstracts": []
            }
            # Find all the references in the related works
            related_work_in_text_citations = re.findall("\[[0-9, ]+\]", related_work_section)
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
                ref_num = re.findall("\[([0-9]+)\]", reference)
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
    #pprint.pprint(rw_dataset)
    with open(os.path.join(reference_data_path, "rw_dataset.json"), "w") as f:
        json.dump(rw_dataset, f)
    orig_df.to_csv(os.path.join(reference_data_path, "new_orig.csv"), index=False)


def extract_citation_order(paper_ref_dict:dict):
    related_work = paper_ref_dict["related_work"]
    citations = re.findall(r"\[[0-9, ]+\]", related_work)
    in_text_ref_dict_order = []
    for citation in citations:
        in_text_ref_dict_order_i = ast.literal_eval(citation)
        in_text_ref_dict_order_i = [int(i) for i in in_text_ref_dict_order_i]
        in_text_ref_dict_order.append(in_text_ref_dict_order_i)
    print(in_text_ref_dict_order)
    return in_text_ref_dict_order


#add_titles_to_orig_df()

#add_related_work_refs_to_orig_df()
#df1 = pd.read_csv(os.path.join(DATA_DIR, "reference_data\\new_orig.csv"))
#for i in df1.index:
#    if df1.loc[i, "title"] == "":
#        print(df1.loc[i, "title"])

#create_related_work_dataset()
with open(os.path.join(DATA_DIR, "reference_data\\rw_dataset.json"), "r") as f:
    data = json.load(f)

extract_citation_order(data[0])

#df = pd.read_csv(os.path.join(DATA_DIR, "reference_data\\new_orig.csv"))
#print(df["rw_refs_in_text_num"])
