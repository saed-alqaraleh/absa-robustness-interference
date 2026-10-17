from absa_xfer.campaign import build_plan, CORE_CONDITIONS

def test_minimal_campaign_size_and_unique_ids():
    rows=build_plan(["restaurant","laptop"],["FacebookAI/roberta-base"],CORE_CONDITIONS,[13,37,73],{"FacebookAI/roberta-base":{"revision":"abc123"}})
    assert len(rows)==36
    assert len({r["run_id"] for r in rows})==36
    assert {r["model_revision"] for r in rows}=={"abc123"}


def test_clean_eval_uses_canonical_test_file_and_arts_use_evaluation_dir(tmp_path):
    from absa_xfer.campaign import make_train_command
    row = {
        "run_id": "restaurant__demo",
        "dataset": "restaurant",
        "model": "FacebookAI/roberta-base",
        "model_revision": "abc123",
        "training_condition": "Clean-Natural",
        "seed": 13,
    }
    hp = {"max_length": 128, "learning_rate": 2e-5, "train_batch_size": 16, "epochs": 8, "pair_order": "sentence_first"}
    cmd = make_train_command(row, tmp_path, hp, "minimal")
    assert f"--eval Clean={tmp_path / 'data_processed' / 'restaurant' / 'test.jsonl'}" in cmd
    for family in ("RevTgt", "RevNon", "AddDiff"):
        assert f"--eval {family}={tmp_path / 'data_processed' / 'restaurant' / 'evaluation' / (family + '.jsonl')}" in cmd
    assert "evaluation/Clean.jsonl" not in cmd
