from absa_xfer.schemas import AspectExample,OpinionRecord,OpinionSpan
from absa_xfer.transformations import generate_revtgt,generate_revnon,build_aspect_expression_library,generate_adddiff

def ex(sid,sentence,term,start,end,label):
    return AspectExample("restaurant","train","sent1",sentence,term,start,end,label,sid,"cluster1")

def test_revtgt_with_override():
    s="Great food and slow service."
    food=ex("foodid",s,"food",6,10,"positive")
    service=ex("servid",s,"service",20,27,"negative")
    ops={
      "foodid":OpinionRecord("foodid","cluster1",(OpinionSpan(0,5,"Great"),),"x","fixture"),
      "servid":OpinionRecord("servid","cluster1",(OpinionSpan(15,19,"slow"),),"x","fixture")
    }
    y=generate_revtgt(food,[food,service],ops,13,{"great":["awful"]},False)
    assert y.transformed_label=="negative"
    assert y.transformed_sentence.startswith("Awful food")

def test_revnon():
    s="Great food and fast service."
    food=ex("foodid",s,"food",6,10,"positive")
    service=ex("servid",s,"service",20,27,"positive")
    ops={
      "foodid":OpinionRecord("foodid","cluster1",(OpinionSpan(0,5,"Great"),),"x","fixture"),
      "servid":OpinionRecord("servid","cluster1",(OpinionSpan(15,19,"fast"),),"x","fixture")
    }
    y=generate_revnon(food,[food,service],ops,13,{"fast":["slow"]})
    assert "slow service" in y.transformed_sentence.lower()
    assert y.transformed_label=="positive"

def test_adddiff_train_only_library():
    s1="Great food."
    food=ex("foodid",s1,"food",6,10,"positive")
    op1={"foodid":OpinionRecord("foodid","cluster1",(OpinionSpan(0,5,"Great"),),"x","fixture")}
    s2="Slow service."
    service=AspectExample("restaurant","train","sent2",s2,"service",5,12,"negative","servid","cluster2")
    op2=OpinionRecord("servid","cluster2",(OpinionSpan(0,4,"Slow"),),"x","fixture")
    ops={**op1,"servid":op2}
    lib=build_aspect_expression_library([food,service],ops)
    y=generate_adddiff(food,[food],ops,lib,13,1,1)
    assert y.transformed_label=="positive"
    assert "slow service" in y.transformed_sentence.lower()
