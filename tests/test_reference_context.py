from ber.reference_context import owner_features
from ber.text import prepare_record


def test_competing_reference_support_and_numeric_disagreement():
    q=prepare_record(dict(entity_id='S1-1',business_name='Ivenus',business_address='38 Holly Hill Lane',country='US'))
    t=prepare_record(dict(entity_id='S2-1',business_name='Ivenusa',business_address='38 Holly Hill Ln',country='US'))
    no=owner_features(q,t,[])
    yes=owner_features(q,t,[('S1-2','38 Holly Hill Lane','US')])
    assert no[0]==0 and no[6]==0
    assert yes[0]>0 and yes[6]==1
    assert yes[7]==0
    different=owner_features(q,t,[('S1-2','93 Other Road','US')])
    assert different[6]<yes[6]
