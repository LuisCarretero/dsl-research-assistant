import json
from dotenv import load_dotenv
import os
from tqdm import tqdm
import re
import ast
from pyalex import Works
import pandas as pd

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")


def add_titles_to_orig_df():
    reference_data_path = os.path.join(DATA_DIR, "reference_data")

    # Fetch the originial data info
    orig_df = pd.read_csv(os.path.join(reference_data_path, "orig.csv"))
    orig_df["oa_refs_titles"] = None

    # Loop through the references and add the title column
    for i in tqdm(orig_df.index):
        paper = orig_df.loc[i]
        try:
            ref_oaids = ast.literal_eval(paper["refs_oaids_from_dois"])
            # Fetch the works
            works = Works()[ref_oaids]
            #print(works[0])
            orig_df.loc[i, "oa_refs_titles"] = str([work["title"] for work in works])
        except:
            orig_df.loc[i, "oa_refs_titles"] = "[]"
        #break

    orig_df.to_csv(os.path.join(reference_data_path, "new_orig.csv"), index=False)

def create_related_work_references_dataset():
    papers_path = os.path.join(DATA_DIR, "challenge10_batch_1\\CVPR_2024\\Conversions\\opencvf-data\\md")
    reference_data_path = os.path.join(DATA_DIR, "reference_data")

    # Fetch the originial data info
    orig_df = pd.read_csv(os.path.join(reference_data_path, "orig.csv"))
    orig_df["rw_refs_oaids"] = None

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
        if len(related_work_titles) > 0 and len(references_titles) > 0:
            related_work_title = related_work_titles[0]
            references_title = references_titles[0]
            # Find the appropriate section
            related_work_section = ""
            references_section = ""
            #print(sections)
            for section in sections:
                if related_work_title in section:
                    related_work_section = section
                elif references_title in section:
                    references_section = section
            # Find all the references in the related works
            related_work_in_text_citations = re.findall("\[[0-9, ]+\]", related_work_section)
            in_text_citations = []
            for elem in related_work_in_text_citations:
                in_text_citations += ast.literal_eval(elem)
            # From the references, remove the ones that are not in the related work
            references_list = references_section.split("\n")
            related_work_oaids = []
            l = orig_paper["refs_oaids_from_dois"].tolist()
            rw_oaids = ast.literal_eval(l[0]) if len(l) > 0 else []
            rw_oaids = [i.upper() for i in rw_oaids]
            rw_info = refs_df.loc[refs_df["oaid"].isin(rw_oaids)]
            for reference in references_list:
                ref_num = re.findall("\[[0-9]+\]", reference)
                if len(ref_num) > 0:
                    ref_num = ref_num[0]
                    ref_num = ast.literal_eval(ref_num)[0]
                    
                    if ref_num in in_text_citations:
                        for i in rw_info.index:
                            if isinstance(rw_info.loc[i, "title"], str) and rw_info.loc[i, "title"].lower() in reference.lower():
                                related_work_oaids.append(rw_info.loc[i, "oaid"])
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_oaids"] = str(related_work_oaids)
            if len(related_work_oaids) > 0:
                count_used += 1
        else:
            orig_df.loc[orig_df["fname"] == paper.replace(".md", ".txt"), "rw_refs_oaids"] = "[]"
    print(f"Used {count_used}/{count_total} = {count_used/count_total*100}% of papers")
    orig_df.to_csv(os.path.join(reference_data_path, "new_orig.csv"), index=False)

#add_titles_to_orig_df()

#create_related_work_references_dataset()

df = pd.read_csv(os.path.join(DATA_DIR, "reference_data\\new_orig.csv"))
print(df["rw_refs_oaids"])
