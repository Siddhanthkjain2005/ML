import numpy as np
from ber.advanced_features import extra_pair_features,EXTRA_NAMES


def test_alternate_address_separates_number_conflict_from_street_identity():
    a=dict(entity_id='S1-1',business_name='Alpha Inc',business_address='1106 Westmoreland Avenue',country='US')
    b=dict(entity_id='S2-1',business_name='Alpha Inc',business_address='1113 Westmoreland Ave',country='US')
    f=dict(zip(EXTRA_NAMES,extra_pair_features(a,b)))
    assert f['address_without_numbers_exact']==1
    assert f['address_first_number_equal']==0
    assert 0<f['address_first_number_edit']<1
    assert f['name_roman_edit']==1
    b['business_address']='001106 Westmoreland Ave'
    f=dict(zip(EXTRA_NAMES,extra_pair_features(a,b)))
    assert f['address_first_number_equal']==1


def test_missing_fields_are_not_positive_evidence():
    r=dict(entity_id='S1-1',business_name='',business_address='',country='')
    v=extra_pair_features(r,r)
    assert len(v)==len(EXTRA_NAMES)
    assert np.isfinite(v).all() and not np.any(v)
