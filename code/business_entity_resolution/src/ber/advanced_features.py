"""Alternate text/number comparisons; raw text and labels remain untouched."""
import re
from functools import lru_cache
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein,JaroWinkler
from .text import normalize_text,prepare_record
from .romanization import romanize

ALIASES={'road':'rd','street':'st','avenue':'ave','lane':'ln','boulevard':'blvd','drive':'dr','court':'ct','place':'pl','highway':'hwy','apartment':'apt','suite':'ste','floor':'fl','building':'bldg','number':'no','corporation':'corp','incorporated':'inc','limited':'ltd','private':'pvt','company':'co'}
TEXT_NAMES=('roman_edit','roman_jaro','roman_ratio','roman_partial','roman_token_set','roman_token_sort','alternate_edit','alternate_token_set','alternate_token_sort','without_numbers_exact','without_numbers_token_set')
NUMBER_NAMES=('first_number_equal','last_number_equal','first_number_edit','best_number_edit','short_numbers_jaccard','long_numbers_jaccard','number_count_ratio','first_number_length_ratio')
EXTRA_NAMES=tuple(f'{field}_{name}' for field in ('name','address') for name in (*TEXT_NAMES,*NUMBER_NAMES))


@lru_cache(maxsize=30000)
def view(text):
    normalized=normalize_text(text)
    roman=romanize(normalized)
    split=re.sub(r'(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)',' ',roman)
    alternate=' '.join(ALIASES.get(t,t) for t in split.split())
    numbers=tuple(str(int(x)) for x in re.findall(r'\d+',normalized))
    without=' '.join(t for t in alternate.split() if not t.isdecimal())
    return roman,alternate,numbers,without


def jaccard(a,b):
    return len(a&b)/len(a|b) if a or b else 0.


def compare(a,b):
    ra,aa,na,wa=view(a);rb,ab,nb,wb=view(b)
    if ra and rb:
        text=[Levenshtein.normalized_similarity(ra,rb),JaroWinkler.normalized_similarity(ra,rb),fuzz.ratio(ra,rb)/100,fuzz.partial_ratio(ra,rb)/100,fuzz.token_set_ratio(ra,rb)/100,fuzz.token_sort_ratio(ra,rb)/100,Levenshtein.normalized_similarity(aa,ab),fuzz.token_set_ratio(aa,ab)/100,fuzz.token_sort_ratio(aa,ab)/100,float(bool(wa) and wa==wb),fuzz.token_set_ratio(wa,wb)/100 if wa and wb else 0.]
    else:text=[0.]*len(TEXT_NAMES)
    if na and nb:
        numbers=[float(na[0]==nb[0]),float(na[-1]==nb[-1]),Levenshtein.normalized_similarity(na[0],nb[0]),max(Levenshtein.normalized_similarity(x,y) for x in na for y in nb),jaccard({n for n in na if len(n)<=4},{n for n in nb if len(n)<=4}),jaccard({n for n in na if len(n)>=5},{n for n in nb if len(n)>=5}),min(len(na),len(nb))/max(len(na),len(nb)),min(len(na[0]),len(nb[0]))/max(len(na[0]),len(nb[0]))]
    else:numbers=[0.]*len(NUMBER_NAMES)
    return text+numbers


def extra_pair_features(left,right):
    a,b=prepare_record(left),prepare_record(right)
    return np.asarray(compare(a.name.normalized,b.name.normalized)+compare(a.address.normalized,b.address.normalized),dtype=np.float32)
