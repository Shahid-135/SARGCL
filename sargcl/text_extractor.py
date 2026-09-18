"""spaCy-based Textual Hypergraph Node and Hyperedge Extraction."""

from typing import List, Tuple

import spacy


class TextNodeExtractor:
    """Extract syntactically grounded text nodes and hyperedges.

    Uses spaCy dependency parsing to identify content-bearing tokens
    (nouns, proper nouns, verbs, adjectives) as nodes, and verb-centred
    predicate-argument groups as hyperedges capturing higher-order
    relational structure.

    Args:
        model: spaCy pipeline name (default ``"en_core_web_sm"``).
    """

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found. "
                f"Please run: python -m spacy download {model}"
            )

    def extract_nodes_and_hyperedges(
        self, texts: List[str]
    ) -> Tuple[List[List[str]], List[List[List[int]]]]:
        """Parse each text into nodes and hyperedges.

        Args:
            texts: List of raw text strings.

        Returns:
            all_nodes: Per-text list of node label strings.
            all_hyperedges: Per-text list of hyperedges, where each
                hyperedge is a sorted list of node indices.
        """
        all_nodes, all_hyperedges = [], []

        for s in texts:
            doc = self.nlp(s)
            nodes, node_idx = [], {}

            for tok in doc:
                if (
                    tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}
                    and not tok.is_stop
                    and tok.text not in node_idx
                ):
                    node_idx[tok.text] = len(nodes)
                    nodes.append(tok.text)

            hedges = []
            for tok in doc:
                if tok.pos_ == "VERB":
                    group = set()
                    if tok.text in node_idx:
                        group.add(node_idx[tok.text])
                    for child in tok.children:
                        if (
                            child.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj"}
                            and child.text in node_idx
                        ):
                            group.add(node_idx[child.text])
                    if len(group) >= 2:
                        hedges.append(sorted(list(group)))

            # Fallback: if no hyperedges found, connect all nodes
            if not hedges and len(nodes) >= 2:
                hedges = [list(range(len(nodes)))]

            all_nodes.append(nodes)
            all_hyperedges.append(hedges)

        return all_nodes, all_hyperedges
