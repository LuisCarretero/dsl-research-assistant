from .._base import _BaseLiteratureReviewGenerator
from lexrank import LexRank


class LexRankLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    def predict(self, query: str, citations: list[str]) -> str:
        documents = [query]
        sentences = citations.split(".")
        lxr = LexRank(documents)
        summary = lxr.get_summary(sentences, summary_size=20)
        return ".".join(summary)