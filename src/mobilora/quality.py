from __future__ import annotations

from collections import Counter

from mobilora.utils import clamp


class QualityScorer:
    def __init__(self) -> None:
        self.metric_name = "token_f1"
        self._bert_score = None
        try:
            from bert_score import score as bert_score  # type: ignore

            self._bert_score = bert_score
            self.metric_name = "bertscore_f1"
        except Exception:
            self._bert_score = None

    def compare(self, reference: str, candidate: str) -> float:
        if self._bert_score is not None:
            precision, recall, f1 = self._bert_score([candidate], [reference], lang="en", verbose=False)
            return float(f1.mean().item())
        return self._token_f1(reference, candidate)

    def _token_f1(self, reference: str, candidate: str) -> float:
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        if not reference_tokens and not candidate_tokens:
            return 1.0
        if not reference_tokens or not candidate_tokens:
            return 0.0
        reference_counts = Counter(reference_tokens)
        candidate_counts = Counter(candidate_tokens)
        overlap = sum(min(reference_counts[token], candidate_counts[token]) for token in reference_counts)
        precision = overlap / max(len(candidate_tokens), 1)
        recall = overlap / max(len(reference_tokens), 1)
        if precision + recall == 0:
            return 0.0
        return clamp(2.0 * precision * recall / (precision + recall), 0.0, 1.0)
