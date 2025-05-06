from writing_tools import LitLLMLiteratureReviewGenerator, HFClientInferenceModel, LexRankLiteratureReviewGenerator, HFLocalInferenceModel
from writing_tools.inference_models import OllamaInferenceModel
from writing_tools._base import _BaseLiteratureReviewGenerator
from dotenv import load_dotenv
import os
import re
import ast
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from rouge_score.scoring import Score
import numpy as np

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")
CITATION_DIR = os.environ.get("CITATION_DIR")

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
    test_data = []

    data_path = os.path.join(DATA_DIR, "challenge10_batch_1\\CVPR_2024\\Conversions\\opencvf-data\\md")

    for paper_filename in tqdm(os.listdir(CITATION_DIR)):
        paper_path = os.path.join(data_path, paper_filename.replace(".json", ".md"))
        citation_path = os.path.join(CITATION_DIR, paper_filename)
        with open(paper_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(citation_path, "r") as c:
            citations_dict = json.loads(c.read())
        processed = process_paper(text)

        citation_references = {}
        for key in citations_dict["references"]:
            citation_references[int(key)] = citations_dict["references"][key]

        abstract, related_work = None, None
        for section in processed["sections"].keys():
            if "Abstract" in section:
                abstract = processed["sections"][section]
            if "Related Work" in section:
                related_work = processed["sections"][section]
        #related_works = processed["sections"]["2. Related Work"] if "2. Related Work" in processed["sections"] else None
        if related_work is not None and abstract is not None:
            # Extract all the citations
            in_text_citation_order = re.findall("\[[0-9, ]+\]", related_work)
            in_text_citation_order = [ast.literal_eval(citation) for citation in in_text_citation_order] # Convert references to list

            cleaned_in_text_citation_order = []
            reference_ids = set()
            for l in in_text_citation_order:
                cleaned_l = []
                for c in l:
                    if c in citation_references.keys():
                        cleaned_l.append(c)
                        reference_ids.add(c)
                if len(cleaned_l) > 0:
                    cleaned_in_text_citation_order.append(cleaned_l)

            reference_ids = sorted(list(reference_ids))
        
            #references = []
            reference_abstracts = [citation_references[i]["abstract"] for i in reference_ids]
            #reference_ids = list(citation_references.keys())

            #for l in cleaned_in_text_citations:
            #    d = {}
            #    for c in l:
            #        d[c] = citation_references[c]
            #    references.append(d)

            if len(cleaned_in_text_citation_order) > 0 and abstract.replace("\n", "") != "":
                data_point = {
                    "abstract": abstract,
                    "reference_abstracts": reference_abstracts,
                    "reference_ids": reference_ids,
                    "in_text_citation_order": cleaned_in_text_citation_order,
                    "related_work": related_work,
                    "paper_name": paper_filename.replace(".json", "")
                }

                test_data.append(data_point)
    return test_data


def evaluate_literature_review_generator(model:_BaseLiteratureReviewGenerator, data, metrics=["bleu"]):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    scores = None

    predictions = []

    for paper in tqdm(data):
        query = paper["abstract"]
        reference_abstracts = paper["reference_abstracts"]
        truth = paper["related_work"]
        prediction = model.predict(query, reference_abstracts, citation_ids=paper["reference_ids"])

        predictions.append(prediction)

        #total_bleu += sentence_bleu([truth.split()], prediction.split())
        scores_new = scorer.score(truth, prediction)
        if scores is None:
            scores = scores_new
        else:
            for key in scores:
                scores[key] = Score(
                    scores_new[key].precision + scores[key].precision,
                    scores_new[key].recall + scores[key].recall,
                    scores_new[key].fmeasure + scores[key].fmeasure
                )
                #print(scores[key])# += scores_new[key]
    for key in scores:
        scores[key] = Score(
            scores[key].precision / len(data),
            scores[key].recall / len(data),
            scores[key].fmeasure / len(data)
        )
    return scores, predictions


if __name__ == "__main__":
    import pprint
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

    data = create_test_data()
    #print(data[0]["reference_ids"])
    #print(data[0]["in_text_citation_order"])

    #rouge, predictions = evaluate_literature_review_generator(LexRankLiteratureReviewGenerator(), data[0:10])
    #inference_model = OllamaInferenceModel()
    #inference_model.set_default_call_kwargs(model="deepseek-r1:7B")
    HF_KEY = os.environ.get("HF_KEY")

    #inference_model = HFClientInferenceModel(provider="novita", api_key=HF_KEY)
    #inference_model.set_default_call_kwargs(model="deepseek-ai/DeepSeek-R1")
    inference_model = OllamaInferenceModel()
    inference_model.set_default_call_kwargs(model="deepseek-r1:7B")
    def postprocess(out:str):
        thoughts = re.findall("<think>[\S\s]*</think>", out)
        for thought in thoughts:
            out = out.replace(thought, "")
        #print(thoughts)
        return out
    inference_model.output_postprocess = postprocess
    #inference_model = HFLocalInferenceModel(model="E:\\ETH MSc Data Science\\Data Science Lab\\Models\\deepseek-r1-distill-qwen-1.5B\\model", 
    #                                        tokenizer="E:\\ETH MSc Data Science\\Data Science Lab\\Models\\deepseek-r1-distill-qwen-1.5B\\tokenizer", 
    #                                        device="cuda",
    #                                        max_new_tokens=np.inf)
    #inference_model.set_default_call_kwargs(max_tokens=np.inf)
    rouge, predictions = evaluate_literature_review_generator(LitLLMLiteratureReviewGenerator(inference_model, method="vanilla"), data[0:1])
    #pprint.pprint(data[0]["reference_abstracts"])
    print(predictions[0])
    #print(data[0]["in_text_citation_order"])
    #pprint.pprint(rouge)
    #print(predictions[0])