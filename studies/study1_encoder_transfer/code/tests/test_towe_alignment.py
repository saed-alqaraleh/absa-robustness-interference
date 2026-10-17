
from absa_xfer.schemas import AspectExample
from absa_xfer.towe import (
    _parse_annotated_field,
    align_towe_to_examples,
    TOWERow,
)


def test_towe_token_tag_parser():

    value = (
        r"The\O food\B is\O "
        r"very\O good\O .\O"
    )

    tokens, tags = (
        _parse_annotated_field(
            value
        )
    )

    assert tokens == [
        "The",
        "food",
        "is",
        "very",
        "good",
        ".",
    ]

    assert tags == [
        "O",
        "B",
        "O",
        "O",
        "O",
        "O",
    ]


def test_towe_alignment_handles_punctuation_and_contraction():

    sentence = (
        "To be completely fair, "
        "the only redeeming factor was the food, "
        "which was above average, "
        "but couldn't make up for all the other "
        "deficiencies of Teodora."
    )

    aspect = "food"

    ex = AspectExample(
        domain="restaurant",
        split="train",
        sentence_id="x",
        sentence=sentence,
        aspect_term=aspect,
        aspect_from=sentence.index(aspect),
        aspect_to=sentence.index(aspect) + len(aspect),
        label="positive",
        source_instance_id="source-x",
        cluster_id="cluster-x",
    )

    tokenized = (
        "To be completely fair , the only redeeming "
        "factor was the food , which was above average , "
        "but could n't make up for all the other "
        "deficiencies of Teodora ."
    )

    tokens = tokenized.split()

    target_tags = [
        "O"
        for _ in tokens
    ]

    target_tags[
        tokens.index("food")
    ] = "B"

    opinion_tags = [
        "O"
        for _ in tokens
    ]

    above = tokens.index(
        "above"
    )

    opinion_tags[above] = "B"
    opinion_tags[above + 1] = "I"

    row = TOWERow(
        sentence=tokenized,
        tokens=tokens,
        target_tags=target_tags,
        opinion_tags=opinion_tags,
        row_id="1",
    )

    aligned, report = (
        align_towe_to_examples(
            [ex],
            [row],
            provenance="test",
        )
    )

    assert (
        report[0]["status"]
        == "matched"
    )

    rec = aligned[
        "source-x"
    ]

    assert len(rec.spans) == 1

    assert (
        rec.spans[0].text
        == "above average"
    )


def test_multiword_target_alignment():

    sentence = (
        "I charge it at night and skip taking "
        "the cord with me because of the good "
        "battery life."
    )

    aspect = "battery life"

    ex = AspectExample(
        domain="laptop",
        split="train",
        sentence_id="x",
        sentence=sentence,
        aspect_term=aspect,
        aspect_from=sentence.index(aspect),
        aspect_to=sentence.index(aspect) + len(aspect),
        label="positive",
        source_instance_id="source-x",
        cluster_id="cluster-x",
    )

    tokenized = (
        "I charge it at night and skip taking "
        "the cord with me because of the good "
        "battery life ."
    )

    tokens = tokenized.split()

    target_tags = [
        "O"
        for _ in tokens
    ]

    b = tokens.index(
        "battery"
    )

    target_tags[b] = "B"
    target_tags[b + 1] = "I"

    opinion_tags = [
        "O"
        for _ in tokens
    ]

    opinion_tags[
        tokens.index("good")
    ] = "B"

    row = TOWERow(
        sentence=tokenized,
        tokens=tokens,
        target_tags=target_tags,
        opinion_tags=opinion_tags,
        row_id="1",
    )

    aligned, report = (
        align_towe_to_examples(
            [ex],
            [row],
            provenance="test",
        )
    )

    assert (
        report[0]["status"]
        == "matched"
    )

    assert (
        aligned["source-x"]
        .spans[0]
        .text
        == "good"
    )
