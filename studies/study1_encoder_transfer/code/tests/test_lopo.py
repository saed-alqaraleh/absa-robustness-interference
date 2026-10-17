from absa_xfer.schemas import AspectExample,OpinionRecord,OpinionSpan
from absa_xfer.materialize import materialize_lopo_conditions


def _fixture():
    sent1="Tasty burgers and crispy fries."
    c1="c1"
    a1=AspectExample("restaurant","train","s1",sent1,"burgers",6,13,"positive","a1",c1)
    a2=AspectExample("restaurant","train","s1",sent1,"fries",25,30,"positive","a2",c1)
    sent2="Awful pasta but rude service."
    c2="c2"
    b1=AspectExample("restaurant","train","s2",sent2,"pasta",6,11,"negative","b1",c2)
    b2=AspectExample("restaurant","train","s2",sent2,"service",21,28,"negative","b2",c2)
    ops={
        "a1":OpinionRecord("a1",c1,(OpinionSpan(0,5,"Tasty"),),"fixture","fixture"),
        "a2":OpinionRecord("a2",c1,(OpinionSpan(18,24,"crispy"),),"fixture","fixture"),
        "b1":OpinionRecord("b1",c2,(OpinionSpan(0,5,"Awful"),),"fixture","fixture"),
        "b2":OpinionRecord("b2",c2,(OpinionSpan(16,20,"rude"),),"fixture","fixture"),
    }
    return [a1,a2,b1,b2],ops


def test_lopo_budget_is_total_not_per_family(monkeypatch):
    xs,ops=_fixture()
    # Avoid dependence on the installed WordNet corpus.
    lex={"tasty":["awful"],"crispy":["soggy"],"awful":["tasty"],"rude":["kind"]}
    import absa_xfer.transformations as t
    orig=t.reverse_opinion_text
    def patched(span,key,antonym_lexicon=None): return orig(span,key,lex)
    monkeypatch.setattr(t,"reverse_opinion_text",patched)
    # materialize module imported the symbol directly; patch there too.
    import absa_xfer.materialize as m
    monkeypatch.setattr(m,"generate_revtgt",lambda x,asps,op,seed: t.generate_revtgt(x,asps,op,seed,lex))
    monkeypatch.setattr(m,"generate_revnon",lambda x,asps,op,seed: t.generate_revnon(x,asps,op,seed,lex))
    out=materialize_lopo_conditions(xs,ops,["a1","a2","b1","b2"],7,total_budget=4)
    assert set(out)=={"Clean+RevTgt+RevNon","Clean+RevTgt+AddDiff","Clean+RevNon+AddDiff"}
    for rows in out.values():
        assert sum(r["is_augmented"] for r in rows)==4
