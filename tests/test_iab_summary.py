import pytest
from scripts.summarize_iab_m2 import joined_cascade


def test_cascade_counts_missed_ai_and_real_false_alarms_separately():
    rows=[dict(sample_id=k,split='dev',is_ai=int(k!='r'),source_label='a' if k!='r' else None) for k in ['x','y','r']]
    d=[dict(sample_id=k,classes=['real','ai'],scores=s) for k,s in [('x',[.1,.9]),('y',[.8,.2]),('r',[.2,.8])]]
    a=[dict(sample_id=k,classes=['a'],scores=[1.]) for k in ['x','y']]
    o=[dict(sample_id=k,classes=['a'],match_scores=[.7]) for k in ['x','y']]
    result=joined_cascade(rows,d,a,o)
    assert result['flagged_real']==1 and result['flagged_ai']==1
    assert result['models']['resnet50']['correctly_detected_and_attributed_over_all_ai']==.5
    assert result['models']['resnet50']['attribution_top1_among_flagged_ai']==1
    with pytest.raises(ValueError,match='ID mismatch'):joined_cascade(rows,d,a,o[:1])
