from .._base import _BaseLiteratureReviewGenerator
from lexrank import LexRank


class LexRankLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    def predict(self, query: str, citations: list[str], citation_ids: list[int]) -> str:
        documents = [query]
        sentences = ".".join(citations).split(".")
        lxr = LexRank(documents)
        summary = lxr.get_summary(sentences, summary_size=20)
        prediction = ""
        for sentence in summary:
            prediction += sentence
            for i, abstract in enumerate(citations):
                if sentence in abstract:
                    prediction += f" [{citation_ids[i]}]"
            prediction += ". "
        return prediction