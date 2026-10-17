from absa_xfer.run_signature import compute_run_signature, SIGNATURE_FIELDS

def test_signature_stable():
    m={k:k for k in SIGNATURE_FIELDS}
    a=compute_run_signature(m)
    b=compute_run_signature(dict(reversed(list(m.items()))))
    assert a==b
