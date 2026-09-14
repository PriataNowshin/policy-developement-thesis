"""Train-only vocabularies for AST node attributes."""

from dataclasses import dataclass

from .config import PAD_TOKEN, UNKNOWN_TOKEN


class Vocabulary:
    def __init__(self, values=()):
        self.value_to_id = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
        for value in values:
            self.add(value)

    def add(self, value: str) -> int:
        if value not in self.value_to_id:
            self.value_to_id[value] = len(self.value_to_id)
        return self.value_to_id[value]

    def encode(self, value: str) -> int:
        return self.value_to_id.get(value, self.value_to_id[UNKNOWN_TOKEN])

    def __len__(self) -> int:
        return len(self.value_to_id)

    def to_dict(self) -> dict[str, int]:
        return dict(self.value_to_id)

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "Vocabulary":
        vocabulary = cls()
        vocabulary.value_to_id = dict(values)
        return vocabulary


@dataclass
class Vocabularies:
    node_types: Vocabulary
    fields: Vocabulary
    lexical: Vocabulary

    def to_dict(self) -> dict:
        return {
            "node_types": self.node_types.to_dict(),
            "fields": self.fields.to_dict(),
            "lexical": self.lexical.to_dict(),
        }


def build_train_vocabularies(train_examples: list[dict]) -> Vocabularies:
    """Build AST vocabularies from training examples only."""
    node_types = Vocabulary()
    fields = Vocabulary()
    lexical = Vocabulary()

    for example in train_examples:
        graph = example["graph"]
        for value in graph["node_types"]:
            node_types.add(value)
        for value in graph["fields"]:
            fields.add(value)
        for node_lexemes in graph["lexemes"]:
            for token in node_lexemes:
                lexical.add(token)
    return Vocabularies(node_types, fields, lexical)
