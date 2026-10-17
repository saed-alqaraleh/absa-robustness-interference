from collections import Counter

from absa_xfer.schemas import AspectExample, OpinionRecord, OpinionSpan
from absa_xfer.materialize import materialize_lopo_conditions
import absa_xfer.campaign as campaign


def _fixture():
    sent1 = "Tasty burgers and crispy fries."
    c1 = "c1"
    a1 = AspectExample("restaurant", "train", "s1", sent1, "burgers", 6, 13, "positive", "a1", c1)
    a2 = AspectExample("restaurant", "train", "s1", sent1, "fries", 25, 30, "positive", "a2", c1)
    sent2 = "Awful pasta but rude service."
    c2 = "c2"
    b1 = AspectExample("restaurant", "train", "s2", sent2, "pasta", 6, 11, "negative", "b1", c2)
    b2 = AspectExample("restaurant", "train", "s2", sent2, "service", 21, 28, "negative", "b2", c2)
    ops = {
        "a1": OpinionRecord("a1", c1, (OpinionSpan(0, 5, "Tasty"),), "fixture", "fixture"),
        "a2": OpinionRecord("a2", c1, (OpinionSpan(18, 24, "crispy"),), "fixture", "fixture"),
        "b1": OpinionRecord("b1", c2, (OpinionSpan(0, 5, "Awful"),), "fixture", "fixture"),
        "b2": OpinionRecord("b2", c2, (OpinionSpan(16, 20, "rude"),), "fixture", "fixture"),
    }
    return [a1, a2, b1, b2], ops


def _patch_generation(monkeypatch):
    lex = {"tasty": ["awful"], "crispy": ["soggy"], "awful": ["tasty"], "rude": ["kind"]}
    import absa_xfer.transformations as t
    import absa_xfer.materialize as m

    orig = t.reverse_opinion_text

    def patched(span, key, antonym_lexicon=None):
        return orig(span, key, lex)

    monkeypatch.setattr(t, "reverse_opinion_text", patched)
    monkeypatch.setattr(m, "generate_revtgt", lambda x, asps, op, seed: t.generate_revtgt(x, asps, op, seed, lex))
    monkeypatch.setattr(m, "generate_revnon", lambda x, asps, op, seed: t.generate_revnon(x, asps, op, seed, lex))


def test_lopo_uses_every_selected_source_exactly_once_per_condition(monkeypatch):
    xs, ops = _fixture()
    _patch_generation(monkeypatch)
    selected = ["a1", "a2", "b1", "b2"]
    out = materialize_lopo_conditions(xs, ops, selected, seed=7, total_budget=4)

    for rows in out.values():
        aug = [r for r in rows if r["is_augmented"]]
        ids = [r["source_instance_id"] for r in aug]
        assert len(aug) == 4
        assert Counter(ids) == Counter(selected)


def test_lopo_odd_budget_rotates_the_extra_example_symmetrically(monkeypatch):
    xs, ops = _fixture()
    _patch_generation(monkeypatch)
    selected = ["a1", "a2", "b1"]
    out = materialize_lopo_conditions(xs, ops, selected, seed=7, total_budget=3)

    expected = {
        "Clean+RevTgt+RevNon": {"RevTgt": 2, "RevNon": 1},
        "Clean+RevTgt+AddDiff": {"RevTgt": 1, "AddDiff": 2},
        "Clean+RevNon+AddDiff": {"RevNon": 2, "AddDiff": 1},
    }
    for condition, family_counts in expected.items():
        aug = [r for r in out[condition] if r["is_augmented"]]
        assert Counter(r["source_instance_id"] for r in aug) == Counter(selected)
        assert Counter(r["perturbation_family"] for r in aug) == Counter(family_counts)


def test_lopo_campaign_profile_is_exactly_18_roberta_runs():
    get_design = getattr(campaign, "get_stage_design", None)
    assert callable(get_design), "campaign.get_stage_design is required for a fail-closed stage definition"
    cfg = {
        "data": {"domains": ["restaurant", "laptop"]},
        "models": {
            "minimal_decisive": "FacebookAI/roberta-base",
            "final": ["google-bert/bert-base-uncased", "FacebookAI/roberta-base", "microsoft/deberta-v3-base"],
        },
        "training": {"seeds_minimal": [13, 37, 73], "seeds_final": [13, 37, 73, 101, 137]},
    }
    design = get_design(cfg, "lopo")
    assert design["models"] == ["FacebookAI/roberta-base"]
    assert design["seeds"] == [13, 37, 73]
    assert design["conditions"] == campaign.LOPO_CONDITIONS
    assert 2 * len(design["models"]) * len(design["conditions"]) * len(design["seeds"]) == 18


def test_lopo_train_command_uses_lopo_manifest(tmp_path):
    row = {
        "run_id": "restaurant__lopo_demo",
        "dataset": "restaurant",
        "model": "FacebookAI/roberta-base",
        "model_revision": "abc123",
        "training_condition": "Clean+RevTgt+RevNon",
        "seed": 13,
    }
    hp = {"max_length": 128, "learning_rate": 2e-5, "train_batch_size": 16, "epochs": 8, "pair_order": "sentence_first"}
    cmd = campaign.make_train_command(row, tmp_path, hp, "lopo")
    assert f"--transformation-manifest {tmp_path / 'transformations' / 'materialized' / 'restaurant' / 'LOPO_MANIFEST.json'}" in cmd
    assert f"--train {tmp_path / 'transformations' / 'materialized' / 'restaurant' / 'Clean+RevTgt+RevNon' / 'train.jsonl'}" in cmd


def test_lopo_can_be_composed_from_frozen_single_family_rows_without_regeneration():
    compose = getattr(__import__("absa_xfer.materialize", fromlist=["materialize_lopo_from_frozen_rows"]), "materialize_lopo_from_frozen_rows", None)
    assert callable(compose), "LOPO must support composition from already-frozen family rows"

    selected = ["s1", "s2", "s3", "s4"]
    clean = [{
        "sentence": f"clean {sid}", "aspect_term": "x", "label": "positive",
        "source_instance_id": sid, "cluster_id": f"c_{sid}",
        "is_augmented": False, "perturbation_family": "Clean",
    } for sid in selected]
    family_rows = {}
    for fam in ("RevTgt", "RevNon", "AddDiff"):
        family_rows[fam] = [{
            "sentence": f"{fam} frozen {sid}", "aspect_term": "x", "label": "positive",
            "source_instance_id": sid, "cluster_id": f"c_{sid}",
            "is_augmented": True, "perturbation_family": fam,
            "generation": {"frozen_marker": f"{fam}:{sid}"},
        } for sid in selected]

    out = compose(clean, family_rows, selected, seed=7)
    assert set(out) == set(campaign.LOPO_CONDITIONS)
    for rows in out.values():
        aug = [r for r in rows if r["is_augmented"]]
        assert Counter(r["source_instance_id"] for r in aug) == Counter(selected)
        for row in aug:
            fam = row["perturbation_family"]
            sid = row["source_instance_id"]
            assert row["generation"]["frozen_marker"] == f"{fam}:{sid}"
            assert row["sentence"] == f"{fam} frozen {sid}"
