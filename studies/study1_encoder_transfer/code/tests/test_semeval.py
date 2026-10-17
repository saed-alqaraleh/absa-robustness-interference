from pathlib import Path
from absa_xfer.semeval import parse_semeval_xml

def test_parse_semeval(tmp_path: Path):
    x=tmp_path/"x.xml"
    x.write_text("""<sentences><sentence id="s1"><text>The food is great.</text><aspectTerms>
    <aspectTerm term="food" polarity="positive" from="4" to="8"/>
    <aspectTerm term="great" polarity="conflict" from="12" to="17"/>
    </aspectTerms></sentence></sentences>""")
    rows=parse_semeval_xml(x,"restaurant","train")
    assert len(rows)==1
    assert rows[0].aspect_term=="food"
    assert rows[0].label=="positive"
