from absa_xfer.integrity import integrity_report

def r(sid,cid,split):
    return {"source_instance_id":sid,"cluster_id":cid,"sentence":sid,"label":"positive"}

def test_cluster_leak_blocks():
    rep=integrity_report([r("a","c","train")],[r("b","d","dev")],[r("z","c","test")])
    assert rep["blocking"]
    assert any(x["type"]=="train_test_cluster_overlap" for x in rep["issues"])
