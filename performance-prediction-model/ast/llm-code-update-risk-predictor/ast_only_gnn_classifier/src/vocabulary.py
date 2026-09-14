"""Train-only AST vocabularies."""

from dataclasses import dataclass

from .config import PAD_TOKEN, UNKNOWN_TOKEN


class Vocabulary:
    def __init__(self):
        self.value_to_id = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}

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
    node_types = Vocabulary()
    fields = Vocabulary()
    lexical = Vocabulary()
    for example in train_examples:
        graph = example["graph"]
        for value in graph["node_types"]:
            node_types.add(value)
        for value in graph["fields"]:
            fields.add(value)
        for node_tokens in graph["lexemes"]:
            for token in node_tokens:
                lexical.add(token)
    return Vocabularies(node_types, fields, lexical)
