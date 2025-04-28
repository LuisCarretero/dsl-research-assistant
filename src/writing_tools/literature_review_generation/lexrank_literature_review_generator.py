from .._base import _BaseLiteratureReviewGenerator
from lexrank import LexRank


class LexRankLiteratureReviewGenerator(_BaseLiteratureReviewGenerator):
    def predict(self, query: str, citations: list[dict[int, dict[str, str]]]) -> str:
        documents = [query]
        sentences = []
        for d in citations:
            for c in d.keys():
                sentences += d[c]["abstract"].split(".")
        lxr = LexRank(documents)
        summary = lxr.get_summary(sentences, summary_size=20)
        return ".".join(summary)