from ber.token_translation import fit_translation,translate_name,nonlatin


def test_training_alignment_learns_nonlatin_only(tmp_path):
 pairs=[]
 for _ in range(10):pairs.extend([('alpha foods','अल्फा फूड्स'),('alpha services','अल्फा सेवा'),('beta foods','बीटा फूड्स'),('beta services','बीटा सेवा')])
 pairs=[(*pair,str(i)) for i,pair in enumerate(pairs)]
 m=fit_translation(pairs,tmp_path/'mapping.json')['mappings']
 assert translate_name('अल्फा फूड्स',m)=='alpha foods'
 assert translate_name('Beta Français',m)=='beta français'
 assert nonlatin('अल्फा') and not nonlatin('français')
 assert translate_name('unseen',m)=='unseen'
 assert translate_name('अनजान फूड्स',m)=='अनजान foods'


def test_single_identity_repetition_is_not_distinct_support(tmp_path):
 m=fit_translation([("alpha","अल्फा","S1-1")]*30,tmp_path/"m.json")
 assert not m["mappings"]


def test_single_word_translation_is_not_swallowed_by_null(tmp_path):
 result=fit_translation([('alpha','अल्फा',f'S1-{i}') for i in range(3)],tmp_path/'m.json')
 assert result['mappings']=={'अल्फा':'alpha'}
 assert result['details']['अल्फा']['mapping_training_components']==3


def test_unidentifiable_generic_word_alignment_is_rejected(tmp_path):
 # Without varied contexts there is no evidence whether the foreign token
 # means the proper name or the generic business term.
 result=fit_translation([('alpha services','अल्फा',f'S1-{i}') for i in range(5)],tmp_path/'m.json')
 assert result['mappings']=={}
 assert result['details']['अल्फा']['margin'] < .2
 assert translate_name('अल्फा',result['mappings'])=='अल्फा'


def test_duplicate_identity_cannot_outvote_independent_components(tmp_path):
 pairs=[('beta','बीटा',f'S1-{i}') for i in range(3)]
 pairs += [('alpha','बीटा','one-repeated-identity')]*100
 result=fit_translation(pairs,tmp_path/'m.json')
 assert result['mappings']=={'बीटा':'beta'}
 assert result['pair_count']==4
 assert result['details']['बीटा']['mapping_training_components']==3


def test_mapping_needs_support_for_that_english_token(tmp_path):
 # Foreign-token frequency alone must not authorize a mapping learned from
 # one training identity. Other identities offer incompatible English names.
 pairs=[('alpha','अल्फा','S1-1')]*100
 pairs += [('beta','अल्फा','S1-2'),('gamma','अल्फा','S1-3')]
 result=fit_translation(pairs,tmp_path/'m.json',min_confidence=0,min_margin=0)
 assert result['details']['अल्फा']['distinct_training_components']==3
 assert not result['mappings']


def test_empty_and_latin_only_input_leave_no_dictionary(tmp_path):
 result=fit_translation([('alpha','français','S1-1'),('','अल्फा','S1-2')],tmp_path/'m.json')
 assert not result['mappings']
 assert result['training_component_count']==0


def test_common_unaligned_foreign_filler_is_not_a_business_alias(tmp_path):
 patterns=[('alpha foods','अल्फा फूड्स'),('alpha services','अल्फा सेवा'),('beta foods','बीटा फूड्स'),('beta services','बीटा सेवा')]
 pairs=[(left,right+' ब्रांड',str(i)) for i,(left,right) in enumerate(patterns*5)]
 result=fit_translation(pairs,tmp_path/'m.json')
 assert translate_name('अल्फा फूड्स ब्रांड',result['mappings'])=='alpha foods ब्रांड'
 assert result['details']['ब्रांड']['accepted'] is False
