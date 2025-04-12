from dotenv import load_dotenv
import os
from tqdm import tqdm
from writing_tools._base import _BaseSuggestionGenerator
from writing_tools import SimpleSuggestionGenerator, OllamaInferenceModel
from nltk.translate.bleu_score import sentence_bleu


load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

# Fetch all the paper segments
paper_segments = []

dir = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md")
for f_dir in tqdm(os.listdir(dir), desc="Loading paper segments..."):
    with open(os.path.join(dir, f_dir), "r", encoding="utf8") as f:
        paper = f.read()
        # Find all the segments one by one
        segments = paper.split("##")
        # Only keep the ones that have at least 50 words
        segments_filtered = []
        for segment in segments:
            # Don't include abstract
            if "Abstract" not in segment:
                if len(segment.split(" ")) >= 50:
                    segments_filtered.append(segment)
        paper_segments += segments_filtered

print(paper_segments[1])
print(len(paper_segments))

paper_segment_sentences = [segment.split(".") for segment in paper_segments]
print(len(paper_segment_sentences))
print(paper_segment_sentences[0])

def test_suggestion_generator(suggestion_generator:_BaseSuggestionGenerator, metrics=["bleu", "rouge"]):
    bleu_total = 0
    predictions_total = 0
    for segment_sentences in tqdm(paper_segment_sentences[:1], desc="Computing scores..."):
        for i in tqdm(range(1, len(segment_sentences)-1)):
            given_text = ''.join(segment_sentences[:i])
            suggestion = suggestion_generator.predict(given_text, -1)
            predictions_total += 1
            bleu_total += sentence_bleu([given_text.split()], suggestion.split())
    print(bleu_total/predictions_total)

test_suggestion_generator(SimpleSuggestionGenerator(OllamaInferenceModel()))