from writing_tools import SimpleLiteratureReviewGenerator
from dotenv import load_dotenv
import os
import re
import ast

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

def process_paper(paper_text:str):
    out = {}
    # Remove images
    paper_text = paper_text.replace("<!-- image -->", "")
    # Extract figure captions
    figure_captions = re.findall("\n\nFigure [0-9]+.*\n\n", paper_text)
    # Extract table captions
    table_captions = re.findall("\n\nTable [0-9]+.*\n\n", paper_text)
    # Extract tables
    tables = re.findall("\n\n[|][\s\S]*[|]\n\n", paper_text)

    # Remove everything that was extracted and store it
    out["figure captions"] = {}
    for i in range(len(figure_captions)):
        figure_caption = figure_captions[i]
        out["figure captions"][i] = figure_caption.replace("\n\n", "")
        paper_text = paper_text.replace(figure_caption.replace("\n\n", ""), "")
    out["tables"] = {}
    for i in range(max(len(table_captions), len(tables))):
        table_caption = table_captions[i] if i < len(table_captions) else ""
        table = tables[i] if i < len(tables) else ""
        out["tables"][i] = {"caption": table_caption.replace("\n\n", ""), "content": table.replace("\n\n", "")}
        paper_text = paper_text.replace(table_caption.replace("\n\n", ""), "")
        paper_text = paper_text.replace(table, "")

    # Extract the sections
    sections = paper_text.split("##")
    out["sections"] = {}
    for section in sections:
        title = section[:section.find("\n\n")].strip().replace("\n\n", "")
        #print(title)
        content = section[section.find("\n\n"):]
        out["sections"][title] = content.replace("\n\n", "")

    return out
     

def create_test_data():
    data_path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md")

    for paper_filename in os.listdir(data_path):
        paper_path = os.path.join(data_path, paper_filename)
        with open(paper_path, "r", encoding="utf-8") as f:
            text = f.read()
        processed = process_paper(text)

        related_works = processed["sections"]["2. Related Work"] if "2. Related Work" in processed["sections"] else None
        if related_works is not None:
            # Extract all the citations
            citations = re.findall("\[[0-9, ]+\]", related_works)
            citations = [ast.literal_eval(citation) for citation in citations] # Convert references to list
            print(citations)
            break


if __name__ == "__main__":
    '''
    path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md\\Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.md")
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    out = process_paper(text)
    print(out["tables"][0]["content"])
    for title in out["sections"]:
        print(title+"\n\n"+out["sections"][title])
        print()
    '''

    create_test_data()