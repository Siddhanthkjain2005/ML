"""Supervised training-only token alignment for cross-script business names.

Sparse IBM Model 1 style EM; no external dictionary or business enrichment.
Only non-Latin target tokens are replaced and confidence/support are recorded.
"""
import json,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from .text import normalize_legal_name,normalize_text
from .pipeline import write_json
from .retrieval import _digest


def nonlatin(token):
 return any(c.isalpha() and 'LATIN' not in unicodedata.name(c,'') for c in token)


def name_tokens(text):
 return normalize_legal_name(text).split()


def fit_translation(pairs,output_path,iterations=8,min_support=3,min_confidence=.60,min_margin=.20):
    if iterations < 1 or min_support < 1 or not 0 <= min_confidence <= 1 or not 0 <= min_margin <= 1:
        raise ValueError('Invalid alignment iterations, support or confidence settings')
    # IBM1 estimates P(foreign | English), with each English distribution
    # normalized over foreign tokens. Normalizing P(English | foreign) during
    # EM instead makes the ever-present NULL token a universal explanation.
    # A small fixed NULL prior permits unaligned tokens without assigning
    # half the posterior to NULL in a perfectly aligned one-word corpus.
    null='__NULL__';null_prior=.1
    sentences=[];seen=set();group_counts=Counter();support_groups=defaultdict(set)
    pair_support=defaultdict(set);english_foreign=defaultdict(set)
    for left,right,group in pairs:
        english=tuple(sorted({t for t in name_tokens(left) if not nonlatin(t) and any(c.isalpha() for c in t)}))
        foreign=tuple(sorted({t for t in name_tokens(right) if nonlatin(t)}))
        if not english or not foreign:continue
        key=(group,english,foreign)
        if key in seen:continue
        seen.add(key);english=english+(null,)
        sentences.append((english,foreign,group));group_counts[group]+=1
        for f in foreign:
            support_groups[f].add(group)
            for e in english:
                english_foreign[e].add(f)
                if e!=null:pair_support[f,e].add(group)
    # Sparse support is enough: pairs that never co-occur cannot align.
    probability={e:{f:1/len(fs) for f in fs} for e,fs in english_foreign.items()}

    def expectation():
        counts=defaultdict(Counter)
        for english,foreign,group in sentences:
            # Each identity has unit total sentence weight, independent of its number
            # of aliases, duplicate rows or languages represented.
            weight=1/group_counts[group]
            for f in foreign:
                scores={e:probability[e].get(f,0.)*(null_prior if e==null else 1.) for e in english}
                denom=sum(scores.values())
                if denom:
                    for e in english:counts[e][f]+=weight*scores[e]/denom
        return counts

    for _ in range(iterations):
        for e,values in expectation().items():
            total=sum(values.values());probability[e]={f:c/total for f,c in values.items()}
    # Acceptance confidence is the component-weighted alignment posterior
    # P(English | foreign), not the generative likelihood P(foreign | English).
    posterior=defaultdict(Counter)
    for e,values in expectation().items():
        for f,count in values.items():posterior[f][e]=count
    mappings={};details={}
    for f,counts in posterior.items():
        total=sum(counts.values());values={e:c/total for e,c in counts.items()}
        ranked=sorted(values.items(),key=lambda x:(x[1],x[0]),reverse=True)
        e,confidence=ranked[0];margin=confidence-(ranked[1][1] if len(ranked)>1 else 0.)
        support=len(support_groups[f])
        aligned_support=len(pair_support.get((f,e),()))
        if aligned_support>=min_support and confidence>=min_confidence and margin>=min_margin and e!=null:mappings[f]=e
        details[f]={'distinct_training_components':support,'mapping_training_components':aligned_support,'best':e,'confidence':confidence,'margin':margin,'accepted':f in mappings}
    result={'schema_version':2,'method':'sparse IBM1 P(foreign|English) EM with NULL; component-balanced posterior acceptance; train-fold positives only','null_alignment_prior':null_prior,'iterations':iterations,'min_support':min_support,'min_confidence':min_confidence,'min_margin':min_margin,'pair_count':len(sentences),'training_component_count':len(group_counts),'mappings':mappings,'details':details,'implementation_sha256':_digest(Path(__file__))}
    write_json(output_path,result);return result


def translate_name(text,mapping):
    return ' '.join(mapping.get(t,t) for t in name_tokens(text))
