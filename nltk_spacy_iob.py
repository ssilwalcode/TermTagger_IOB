from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from sklearn.metrics import classification_report
import os 
import pandas as pd
# use spacy tokenizer to ensure identical tokenization
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt')
nltk.download("averaged_perceptron_tagger")
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "This is an example sentence."

#tokenize with spacy
tokenized_sent = [token.text for token in nlp(sentence)]
print(tokenized_sent)

#tokenize with nltk
print(word_tokenize(sentence))

"""-------nltk iob tagging-------"""

# get nltk IOB tags
def nltk_iob(sentence):
  tokenized_sent = pos_tag([token.text for token in nlp(sentence)])
  ner_sent = ne_chunk(tokenized_sent) 
  print(ner_sent)
  iob = tree2conlltags(ner_sent)
  print(iob)
  out = " ".join([item[2][0] for item in iob])
  return [sentence,out]

# sentence = "This article deals with deverbal nominalizations in Spanish ; concretely , we focus on the denotative distinction between event and result nominalizations ."
# nltk_iob(sentence)

test_df = pd.read_csv("test_nltk_spacy.csv") #test file same as gold annotated test file (saved as different name)
print(test_df.head())

# get IOB tags from nltk
nl_iob = []
for line in test_df['sent']:
  nl_iob.append(nltk_iob(line))

# store nltk tags in a df (same format as text_new.csv)
out_df= pd.DataFrame(nl_iob)
out_df = out_df.rename(columns = {0:'sent', 1:'tags'})
out_df.head()

out_df.to_csv("test_nltk.csv", index = False)

# get accuracy scores
target = [tag for tag in " ".join(test_df['tags']).split()]
prediction = [tag for tag in " ".join(out_df['tags']).split()]
print(len(target), len(prediction), len(test_df), len(out_df))
print(classification_report(target, prediction))

"""-------SpaCy IOB-------"""

# get spacy IOB tags
def spacy_iob(sentence):
  out = " ".join([token.ent_iob_ for token in nlp(sentence)])
  return [sentence,out]

# get IOB tags from spacy
spa_iob = []
for line in test_df['sent']:
  spa_iob.append(spacy_iob(line))

# store spacy tags in a df (same format as text_new.csv)
spacy_df = pd.DataFrame(spa_iob)
spacy_df = spacy_df.rename(columns = {0:'sent', 1:'tags'})
spacy_df.head()

spacy_df.to_csv("test_spacy.csv", index = False)

# get accuracy scores
target = [tag for tag in " ".join(test_df['tags']).split()]
prediction = [tag for tag in " ".join(spacy_df['tags']).split()]
print(len(target), len(prediction), len(test_df), len(spacy_df))
print(classification_report(target, prediction))