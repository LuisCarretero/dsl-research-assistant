from dotenv import load_dotenv
import os
from tqdm import tqdm
from writing_tools._base import _BaseSuggestionGenerator
from writing_tools import SimpleSuggestionGenerator, OllamaInferenceModel
from nltk.translate.bleu_score import sentence_bleu
import re
import ast
import json


load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")
CITATION_DIR = os.environ.get("CITATION_DIR")


def compute_paper_bleu(suggestion_generator:_BaseSuggestionGenerator, paper_filename:str):
    path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md\\"+paper_filename+".md")
    citation_path = os.path.join(CITATION_DIR, paper_filename+".json")

    with open(citation_path, "r") as f:
        citation_dict = json.load(f)

    with open(path, "r") as f:
        text = f.read()

    # Get all sections
    sections = text.split("##")
    # Remove the last section with the literature
    text = "##".join(sections[:-1])

    total_bleu = 0
    n_runs = 0
    # Loop through all in-text citations
    for elem in tqdm(list(re.finditer("\[[^]]+\]", text))):
        # Fetch the reference abstracts
        try:
            l = ast.literal_eval(text[elem.start():elem.end()]) # Convert references to list
            truth = ".".join(text[elem.end():].split(".")[:2])
            abstracts = []
            skip = False
            for i in l.keys():
                if str(i) in citation_dict["references"].keys():
                    abstracts.append(citation_dict["references"][str(i)]["abstract"])
                else: 
                    skip = True
            if not skip:
                prediction = suggestion_generator.predict(text, elem.start(), abstracts)
                bleu = sentence_bleu([truth.split()], prediction.split())
                total_bleu += bleu
                n_runs += 1
        except Exception as e:
            print(e)
            continue

    return total_bleu/n_runs


#bleu = compute_paper_bleu(SimpleSuggestionGenerator(OllamaInferenceModel()), "Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper")
#print(f"Bleu: {bleu}")

paper_filename = "Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper"
suggestion_generator = SimpleSuggestionGenerator(OllamaInferenceModel())

path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md\\"+paper_filename+".md")
citation_path = os.path.join(CITATION_DIR, paper_filename+".json")

with open(citation_path, "r") as f:
    citation_dict = json.load(f)

with open(path, "r") as f:
    text = f.read()

# Get all sections
sections = text.split("##")
# Remove the last section with the literature
text = "##".join(sections[:-1])

# Loop through all in-text citations
count = 0
for elem in tqdm(list(re.finditer("\[[^]]+\]", text))):
    # Fetch the reference abstracts
    try:
        l = ast.literal_eval(text[elem.start():elem.end()]) # Convert references to list
        previous_sentences = ".".join(text[:elem.start()].split(".")[:-1])
        truth = text[:elem.start()].split(".")[-1]
        abstracts = []
        for i in l:
            if str(i) in citation_dict["references"].keys():
                abstracts.append(citation_dict["references"][str(i)]["abstract"])
        prediction = suggestion_generator.predict(text, elem.start(), abstracts)
        print(f"Abstracts: {abstracts}")
        print()
        print(f"Previous text: {'.'.join(previous_sentences.split('.')[-10:])}")
        print()
        print(f"Prediction: {prediction}")
        print()
        print(f"Truth: {truth}")
        count += 1
        if count == 2:
            break
    except Exception as e:
        print(e)
        continue