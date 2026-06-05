"""
ner_pipeline.py — Named Entity Recognition using pretrained model
Model: dslim/bert-base-NER (production-grade, trained on CoNLL-2003)
WHY pretrained: NER requires large labeled datasets with BIO tags.
Using a production model is the correct engineering decision here.
Our fine-tuning effort goes into classification where our data adds value.
"""

from transformers import pipeline  # HuggingFace high-level pipeline API
from collections import defaultdict  # for grouping entities by type


# load NER pipeline once at module level
# WHY: loading a transformer model takes ~2-3 seconds
# loading once at import time means all requests after that are fast
# aggregation_strategy="simple" merges subword tokens back into full words
# e.g. "Micro", "##soft" → "Microsoft" with one label
ner_model = None


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Run NER on input text and return entities grouped by type.

    BIO tagging explained:
    B-ORG = Beginning of an Organization entity
    I-ORG = Inside (continuation) of an Organization entity
    O     = Outside any entity (plain text)
    aggregation_strategy="simple" handles this merging automatically.

    Entity types in dslim/bert-base-NER:
    PER = Person names
    ORG = Organizations
    LOC = Locations
    MISC = Miscellaneous (products, events, etc.)

    Returns dict like:
    {
        "PER": ["Elon Musk", "Tim Cook"],
        "ORG": ["Tesla", "Apple Inc"],
        "LOC": ["San Francisco", "India"],
        "MISC": ["iPhone 15"]
    }
    """
    # run the NER model on input text
    # returns list of dicts: [{"entity_group": "ORG", "word": "Tesla", "score": 0.99}, ...]
    raw_entities = ner_model(text)

    # group entities by their type using defaultdict
    # defaultdict(list) means accessing a missing key creates an empty list automatically
    grouped = defaultdict(list)

    for entity in raw_entities:
        entity_type = entity["entity_group"]  # e.g. "ORG", "PER"
        entity_word = entity["word"].strip()  # e.g. "Tesla"
        confidence = entity["score"]          # float between 0 and 1

        # only keep entities with confidence > 0.85 to reduce false positives
        if confidence > 0.85 and entity_word not in grouped[entity_type]:
            # avoid duplicates by checking before appending
            grouped[entity_type].append(entity_word)

    # convert defaultdict to regular dict for clean return
    return dict(grouped)


def extract_entities_from_chunks(chunks: list[str]) -> dict[str, list[str]]:
    global ner_model

    if ner_model is None:
       ner_model = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
       )
    """
    Run NER across all chunks and merge results.
    WHY: NER model has a token limit (~512 tokens).
    Running on full document at once may exceed limit.
    Running on chunks and merging handles any document length.
    """
    merged = defaultdict(set)  # use set to automatically deduplicate across chunks

    for chunk in chunks:
        chunk_entities = extract_entities(chunk)
        for entity_type, words in chunk_entities.items():
            # add each word to the set for that type
            merged[entity_type].update(words)

    # convert sets back to sorted lists for consistent output
    return {k: sorted(list(v)) for k, v in merged.items()}
