from absa_xfer.source_control import select_common_pool

def test_common_pool_budget_exact():
    rows=[]
    for i in range(20):
        rows.append({
          "source_instance_id":str(i),"eligible_all3":True,"domain":"restaurant",
          "label":"positive" if i%2 else "negative","aspect_count_bin":"2",
          "sentence_length_bin":"10-19"
        })
    ids=select_common_pool(rows,7,13)
    assert len(ids)==7
    assert len(set(ids))==7
