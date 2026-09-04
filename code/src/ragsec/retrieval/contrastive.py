import logging

from .base import RetrievedDoc

logger = logging.getLogger(__name__)


class ContrastiveRetriever:
    def __init__(self, mal_retriever, ben_retriever):
        self.mal_retriever = mal_retriever
        self.ben_retriever = ben_retriever

    def retrieve(
        self, query: str, k_mal: int = 3, k_ben: int = 3
    ) -> tuple[list[RetrievedDoc], list[RetrievedDoc]]:
        mal_results = self.mal_retriever.retrieve(query, k=k_mal)
        for doc in mal_results:
            doc.pool = "malicious"
        ben_results = self.ben_retriever.retrieve(query, k=k_ben)
        for doc in ben_results:
            doc.pool = "benign"
        return mal_results, ben_results

    def retrieve_combined(self, query: str, k_mal: int = 3, k_ben: int = 3) -> list[RetrievedDoc]:
        mal, ben = self.retrieve(query, k_mal, k_ben)
        return mal + ben
