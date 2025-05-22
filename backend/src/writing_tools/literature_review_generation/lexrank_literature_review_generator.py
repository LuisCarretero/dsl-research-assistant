from .._base import _BaseLiteratureReviewGenerator
from lexrank import LexRank


class LexRankLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    def predict_entire(self, query:str, references:list[str], reference_ids:list[int]) -> str:
        documents = [query]
        sentences = ".".join(references).split(".")
        lxr = LexRank(documents)
        summary = lxr.get_summary(sentences, summary_size=len(references))
        prediction = ""
        for sentence in summary:
            prediction += sentence
            for i, abstract in enumerate(references):
                if sentence in abstract:
                    prediction += f" [{reference_ids[i]}]"
            prediction += ". "
        return prediction
    
    def predict_next(self, query:str, references:list[str], reference_ids:list[int], related_work_draft:str):
        documents = [query, related_work_draft]
        sentences = ".".join(references).split(".")
        lxr = LexRank(documents)
        summary = lxr.get_summary(sentences, summary_size=len(references))
        prediction = ""
        for sentence in summary:
            prediction += sentence
            for i, abstract in enumerate(references):
                if sentence in abstract:
                    prediction += f" [{reference_ids[i]}]"
            prediction += ". "
        return prediction