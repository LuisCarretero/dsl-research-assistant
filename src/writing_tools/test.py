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

def create_suggestion_test_dataset():
    path = os.path.join(DATA_DIR, "challenge10_batch_1\\CVPR_2024Conversions\\opencvf-data\\md") # Path to the markdown papers
    store_path = "data\\citations" # Path where to store the data

    # Loop through all the papers
    for paper in tqdm(os.listdir(path)):
        d = {}
        # Read the text of the paper
        with open(os.path.join(path, paper), "r", encoding="utf-8") as f:
            text = f.read()
        # Get the References section
        sections = text.split("##")
        references = sections[-1]
        # Get the abstracts of the references
        d["references"] = {}
        # Store the JSON
        fpath = os.path.join(store_path, paper.replace(".md", ".json"))
        # Loop through all the referneces
        for elem in tqdm(re.finditer("\[[^]]+\]", references)):
            # Try to store them using OpenAlex
            try:
                l = ast.literal_eval(references[elem.start():elem.end()]) # Convert references to list
                ref_num = l[0] # Get the citation number
                store_new = True
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        d_old = json.load(f)
                    store_new = store_new and (str(ref_num) not in d_old["references"].keys())
                if store_new: 
                    ref_text = references[elem.end():elem.end()+references[elem.end():].find("\n")] # Get the text of the reference
                    title = ref_text.split(".")[1] # Get the title of the reference
                    work = Works().search(title).get()[0] # Search for the reference with OpenAlex
                    abstract = work["abstract"]
                    # Store the abstract and DOI of the reference number
                    if abstract is not None:
                        d["references"][ref_num] = {}
                        d["references"][ref_num]["abstract"] = abstract
                        d["references"][ref_num]["doi"] = work["doi"]
                else:
                    d["references"] = d_old["references"]
            except:
                continue

        with open(fpath, "w") as out:
            json.dump(d, out)

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

if __name__ == "__main__":
    import numpy as np
    #create_suggestion_test_dataset()
    with open(os.path.join(DATA_DIR, "reference_data\\rw_dataset.json"), "r") as f:
        data = json.load(f)
    lengths = []
    for data_point in data:
        cit_order = extract_citation_order(data_point)
        if len(cit_order) != 0:
            lengths.append(len(cit_order))
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {np.mean(lengths)}")
    """
    from dotenv import load_dotenv
    import os
    from tqdm import tqdm
    import re
    import ast
    from pyalex import Works

    load_dotenv()

    DATA_DIR = os.environ.get("DATA_DIR")

    path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md")
    
    #count = 0
    #total = 0
    #no_related_work = []
    #for f_name in tqdm(os.listdir(path)):
    #    with open(os.path.join(path, f_name), "r", encoding="utf-8") as f:
    #        text = f.read()
    #        count += "Related Work" in text
    #        if "Related Work" not in text:
    #            no_related_work.append(f_name)
    #        total += 1
    #
    #print(f"Contains 'Related Work' in the text: {count}/{total}, which is {count/total}%")
    #print(f"Papers without 'Related Work': {no_related_work}")

    paper = "Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.md"

    with open(os.path.join(path, paper), "r", encoding="utf-8") as f:
        text = f.read()

    # Remove images
    text = text.replace("<!-- image -->", "")
    # Remove figure text
    paragraphs = text.split("\n\n")
    text = ""
    for paragraph in paragraphs:
        if paragraph[:6] != "Figure":
            text = "\n\n".join([text, paragraph])

    #print(text)

    # Get the Related Work section
    sections = text.split("##")
    related_work = ""
    references = ""
    for section in sections:
        print(section[:section.find("\n")])
        if "Related Work" in section.strip()[:section.find("\n")]:
            related_work = section
        if "References" in section.strip()[:section.find("\n")]:
            references = section
            #break

    # Get the abstracts of the references
    citation_abstracts = {}
    for elem in re.finditer("\[[^]]+\]", references):
        try:
            l = ast.literal_eval(references[elem.start():elem.end()])
            ref_num = l[0]
            ref_text = references[elem.end():elem.end()+references[elem.end():].find("\n")]
            title = ref_text.split(".")[1]
            print(f"[{ref_num}] {ref_text} | {title}")
            #print()
            citation_abstracts[ref_num] = Works().search(title).get()[0]["abstract"]
            #break
        except Exception as e:
            print(e)

    for k in citation_abstracts:
        print(citation_abstracts[k])

    # Find all the in-text citations (looking for [number, number, ...])

    for elem in re.finditer("\[[^]]+\]", related_work):
        #print(elem)
        #print(text[elem.start():elem.end()])
        try:
            l = ast.literal_eval(text[elem.start():elem.end()])
            print(l)
        except:
            continue
    """

