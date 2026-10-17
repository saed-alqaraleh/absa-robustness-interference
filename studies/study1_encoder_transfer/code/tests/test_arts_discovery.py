from pathlib import Path
from absa_xfer.arts_discovery import infer_family_from_path, discover_arts_files

def test_family_hints_and_safe_mapping(tmp_path: Path):
    for name in ["restaurant_RevTgt.jsonl","restaurant_RevNon.jsonl","restaurant_AddDiff.jsonl"]:
        (tmp_path/name).write_text('{}\n',encoding='utf-8')
    assert infer_family_from_path("x/rev_tgt/test.json") == "RevTgt"
    rep=discover_arts_files(tmp_path)
    assert rep["n_candidate_files"]==3
    assert rep["automatic_family_mapping_safe"] is True
