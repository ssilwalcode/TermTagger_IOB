import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
nltk.download('maxent_ne_chunker')

bert_df = pd.read_csv("predicted_BERT.csv")
#bert_df.head()

test_nltk = pd.read_csv( "test_nltk.csv")
test_df = pd.read_csv("test_new2.csv")

from pandas.compat import np_version_under1p20
def ner_span(tags):
  ner_idx = set()
  tags = tags.split()
  span_start = span_end = -1
  for i, tag in enumerate(tags):
    if tag == "B":
      span_start = i
      span_end = i
      if i == len(tags) -1 or tags[i+1] != "I": # next one is O or B or end of string
        ner_idx.add((span_start, span_end))
        
    elif tag == "I":
      span_end = i
      if tags[i-1] == "O":
        span_start = i
      if i == len(tags) -1 or tags[i+1] != "I":
        ner_idx.add((span_start, span_end))
    

  return ner_idx

test_df['tag_span'] = test_df['tags'].apply(lambda x: ner_span(str(x)))
test_nltk['tag_span'] = test_nltk['tags'].apply(lambda x: ner_span(str(x)))

bert_df['tag_span'] = bert_df['true_labels'].apply(lambda x: ner_span(str(x)))

def get_score(ref_df, pred_df):
  gold_ner_span = ref_df.apply(lambda x: ner_span(str(x)))
  pred_ner_span = pred_df.apply(lambda x: ner_span(str(x)))
  ner_gold_counts = 0
  ner_pred_counts = 0
  ner_tp = 0
  for ner_gold, ner_pred in zip(gold_ner_span, pred_ner_span):
    ner_tp += len(set(ner_gold).intersection(set(ner_pred)))
    ner_gold_counts += len(ner_gold)
    ner_pred_counts += len(ner_pred)
  recall =  ner_tp / ner_gold_counts
  precision = ner_tp / ner_pred_counts
  f1 = 2*(recall * precision) / (recall + precision)
  return print("precision: ", round(precision,3), "\nrecall: ", round(recall, 3), "\nF1: ", round(f1, 3))

test_spacy = pd.read_csv("test_spacy.csv")

"""-------error analysis---------"""

import itertools

def err_span(ref_tags, pred_tags, return_spans = True):
  ref_spans = sorted(ner_span(ref_tags))
  pred_spans = sorted(ner_span(pred_tags))
  #print(ref_spans, pred_spans)
  ref_errs = set(ref_spans) - set(pred_spans)
  pred_errs = set(pred_spans) - set(ref_spans)
  combs = list(itertools.product(ref_errs, pred_errs))
  
  err_dict = {"gold total": ref_spans, "pred total": pred_spans, "correct": set(), "missing entities": set(), "wrong tag": set(), "wrong range": [], "extra wrong range":[], "extra no match": set()}
  err_count_dict = {"gold total": len(ref_spans), "pred total": len(pred_spans), "correct": 0, "missing entities": 0, "wrong tag": 0, "wrong range": 0, "extra wrong range":0, "extra no match": 0}

  matched_ref_pred = {ref_span:set() for ref_span in ref_errs}
  matched_pred_ref = set()

  #get entities that have overlapping but not identical spans in pred and ref
  for (ref_st, ref_end), (pred_st, pred_end) in sorted(combs):
    ref_span = range(ref_st, ref_end + 1)
    pred_span = range(pred_st, pred_end + 1)
    #print(ref_span, pred_span)
    overlap = set(ref_span).intersection(pred_span)
    if overlap:
      if len(matched_ref_pred[(ref_st, ref_end)]) != 0:
        err_count_dict['extra wrong range'] += 1
      
      matched_ref_pred[(ref_st, ref_end)].add((pred_st, pred_end))
      err_dict['wrong range'].append(set([(pred_st, pred_end)]))
      
      matched_pred_ref.add((pred_st, pred_end))
      


  # entities only in ref but no overlapping ranges in pred    
  err_dict['missing entities'] = sorted([r for r, p in matched_ref_pred.items() if p == set()])

  # entities that are only in pred but not in ref, or when there are more than one pred_span for each ref_span
  #err_dict['extra wrong range'] = sum([len(p) -1 for r, p in matched_ref_pred.items() if len(p) > 1])
  err_dict['extra no match'] = [p for p in pred_errs if p not in matched_pred_ref]

  # entities that have correct spans but wrong tag
  correct_spans = set(ref_spans).intersection(set(pred_spans))
  for span in correct_spans:
    st = span[0]
    end = span[1]
    for idx in range(st, end+1):
      #print(idx)
      if ref_tags.split()[idx] == pred_tags.split()[idx]:
        continue 
      else:
        err_dict['wrong tag'].add((st, end))
  
  # get entities that match in both range and tag
  err_dict['correct'] = correct_spans - err_dict['wrong tag']

  for k, v in err_dict.items():
    if type(v) == int:
      l = v
      v = ""
      err_count_dict[k] = l
    else:
      err_dict[k] = sorted(err_dict[k])
      v = err_dict[k]
      l = len(v)
      if err_count_dict[k] == 0:
        err_count_dict[k] = l
      
  if return_spans:
    return err_dict
  else:
    return [v for v in err_count_dict.values()]

def get_errors(ref_df, pred_df):
  err_df = pd.DataFrame({"ref":ref_df, "pred":pred_df})
  err_df[["gold total", "pred total", "correct", "missing", "wrong tag", "wrong range","extra wrong range","extra no match"]] = err_df.apply(lambda x: err_span(x.ref, x.pred, return_spans = False), axis = 1).values.tolist()
  return err_df

nltk_errors = get_errors(test_df['tags'], test_nltk['tags'])

#check that the error counts add up 
nltk_errors['check'] = nltk_errors['correct'] + nltk_errors['missing']+ nltk_errors['wrong range'] - nltk_errors['extra wrong range'] + nltk_errors['wrong tag']== nltk_errors['gold total']

spacy_errors = get_errors(test_df['tags'], test_spacy['tags'])
spacy_errors['check'] = spacy_errors['correct'] + spacy_errors['missing']+ spacy_errors['wrong range'] - spacy_errors['extra wrong range'] + spacy_errors['wrong tag']== spacy_errors['gold total']

bert_errors = get_errors(bert_df['true_labels'], bert_df['predicted_labels'])
bert_errors['check'] = bert_errors['correct'] + bert_errors['missing']+ bert_errors['wrong range'] - bert_errors['extra wrong range'] + bert_errors['wrong tag']== bert_errors['gold total']