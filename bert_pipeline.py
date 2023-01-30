# pip install transformers
import pandas as pd
import re
import os
import csv
import torch 
import copy
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import accuracy_score, classification_report
import random

"""------bert-NER method------"""

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

# Cleaning train data due to tags issue

train_data['tags'] = train_data['tags'].replace("b", "B", regex=True)
train_data['tags'] = train_data['tags'].replace("II", "I", regex=True)

# Cleaning test data due to tags issue

test_data['tags'] = test_data['tags'].replace("o", "O", regex=True)
test_data['tags'] = test_data['tags'].replace("0", "O", regex=True)

len(train_data)
train_data.head()

# checking
print(train_data.iloc[4].sent)
print(train_data.iloc[4].tags)

# getting the tags

tags = [i.split() for i in train_data['tags'].values.tolist()]
unique_tags = set()

for t in tags:
  [unique_tags.add(i) for i in t if i not in unique_tags]

tags_to_ids = {k: v for v, k in enumerate(unique_tags)}
ids_to_tags = {v: k for v, k in enumerate(unique_tags)}
print(tags_to_ids)
print(ids_to_tags)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class dataset(Dataset):
  def __init__(self, df, tokenizer, max_len):
        self.len = len(df)
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        sentence = self.data.sent[index].strip().split()  
        word_tags = self.data.tags[index].split(" ") 

        # "return_offsets_mapping" provides us with the start and end of a token divided into subtokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        tags = [tags_to_ids[tag] for tag in word_tags]
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        encoded_tags = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            encoded_tags[idx] = tags[i]
            i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['tags'] = torch.as_tensor(encoded_tags)
        
        return item

  def __len__(self):
        return self.len

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1

training_set = dataset(train_data, tokenizer, MAX_LEN)
testing_set = dataset(test_data, tokenizer, MAX_LEN)
validation_set = dataset(dev_data, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
validation_loader = DataLoader(validation_set, **val_params)

bert_model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tags_to_ids))
bert_model.to(device)

EPOCHS = 7
LEARNING_RATE = 1e-04
MAX_GRAD_NORM = 10

# code partly based on https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L344

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def train_val(model, train, valid, epochs, learning_rate, grad_norm, verbose=True):
  
  model_tr = copy.deepcopy(model)

  optimizer = torch.optim.Adam(params=model_tr.parameters(), lr= learning_rate)

  all_val_acc = 0

  for epoch in range(epochs):
    tr_loss, tr_accuracy = 0, 0
    tr_preds, tr_tags = [], []

    model_tr.train(True)

    for idx, batch in enumerate(train):
      train_ids = batch['input_ids'].to(device, dtype = torch.long)
      train_mask = batch['attention_mask'].to(device, dtype = torch.long)
      train_tags = batch['tags'].to(device, dtype = torch.long)

      optimizer.zero_grad()

      train_loss, train_logits = model_tr(input_ids=train_ids, attention_mask=train_mask, labels=train_tags, return_dict=False)
      tr_loss += train_loss.item()

      #computing training accuracy
      tr_flat_targets = train_tags.view(-1)
      tr_active_logits = train_logits.view(-1, model_tr.num_labels)
      tr_flat_predicts = torch.argmax(tr_active_logits, axis=1)

      tr_active_acc = train_tags.view(-1) != -100

      train_tags = torch.masked_select(tr_flat_targets, tr_active_acc)
      train_predicts = torch.masked_select(tr_flat_predicts, tr_active_acc)

      tr_tags.extend(train_tags)
      tr_preds.extend(train_predicts)

      current_tr_acc = accuracy_score(train_tags.cpu().numpy(), train_predicts.cpu().numpy())
      tr_accuracy += current_tr_acc

      # gradient clipping
      torch.nn.utils.clip_grad_norm_(
          parameters=model_tr.parameters(), max_norm=grad_norm
      )

      train_loss.backward()
      optimizer.step()
    
    model_tr.train(False)

    val_loss, val_accuracy = 0, 0
    val_preds, val_tags = [], []

    for idx, batch in enumerate(valid):
      valid_ids = batch['input_ids'].to(device, dtype=torch.long)
      valid_mask = batch['attention_mask'].to(device, dtype=torch.long)
      valid_tags = batch['tags'].to(device, dtype=torch.long)

      valid_loss, valid_logits = model_tr(input_ids=valid_ids, attention_mask=valid_mask, labels=valid_tags, return_dict=False)
      val_loss += valid_loss.item()

      val_flat_targets = valid_tags.view(-1)
      val_active_logits = valid_logits.view(-1, model_tr.num_labels)
      val_flat_predicts = torch.argmax(val_active_logits, axis=1)

      val_active_acc = valid_tags.view(-1) != -100

      valid_tags = torch.masked_select(val_flat_targets, val_active_acc)
      valid_predict = torch.masked_select(val_flat_predicts, val_active_acc)

      val_tags.extend(valid_tags)
      val_preds.extend(valid_predict)

      current_val_acc = accuracy_score(valid_tags.cpu().numpy(), valid_predict.cpu().numpy())
      val_accuracy += current_val_acc


    tr_ep_loss = tr_loss / len(train)
    tr_acc = tr_accuracy / len(train)

    val_ep_loss = val_loss / len(valid)
    val_acc = val_accuracy / len(valid)

    print('======== Training/Validation result, EPOCH {:} / {:} ========'.format(epoch+1, epochs))
    print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(tr_ep_loss, tr_acc))
    print('Valid Loss: {:.4f}, Valid Acc: {:.4f}'.format(val_ep_loss, val_acc))

    if val_acc > all_val_acc:
      all_val_acc = val_acc
      model_tr.save_pretrained('./best_model/')

train_val(bert_model, training_loader, validation_loader, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM)

def test_loop(model_path, test):

  model_test = BertForTokenClassification.from_pretrained(model_path)
  model_test.to(device)

  model_test.eval()

  t_loss, t_accuracy = 0, 0
  test_preds, test_tags = [], [] #for storing sequence output
  out_pred, out_tags = [], [] 

  with torch.no_grad():
    for idx, batch in enumerate(test):
      test_ids = batch['input_ids'].to(device, dtype=torch.long)
      test_mask = batch['attention_mask'].to(device, dtype=torch.long)
      test_tag = batch['tags'].to(device, dtype=torch.long)

      test_loss, test_logits = model_test(input_ids=test_ids, attention_mask=test_mask, labels=test_tag, return_dict=False)

      t_loss += test_loss.item()

      t_flat_targets = test_tag.view(-1)
      t_active_logits = test_logits.view(-1, model_test.num_labels)
      t_flat_predicts = torch.argmax(t_active_logits, axis=1)

      t_active_acc = test_tag.view(-1) != -100

      t_tags = torch.masked_select(t_flat_targets, t_active_acc)
      t_predict = torch.masked_select(t_flat_predicts, t_active_acc)

      out_pred.extend(t_predict)
      out_tags.extend(t_tags)
      #for csv
      test_tags.append(t_tags)
      test_preds.append(t_predict)

      current_test_acc = accuracy_score(t_tags.cpu().numpy(), t_predict.cpu().numpy())
      t_accuracy += current_test_acc

  #used for classification report
  t_tags = [ids_to_tags[id.item()] for id in out_tags]
  t_predict = [ids_to_tags[id.item()] for id in out_pred]
  #getting sequence output for csv
  out_t = [' '.join([ids_to_tags[id.item()] for id in te]) for te in test_tags]
  out_p = [' '.join([ids_to_tags[id.item()] for id in tep]) for tep in test_preds]


  t_ep_loss = t_loss / len(test)
  t_ep_acc = t_accuracy / len(test)


  print('=========== Test results ===+=======')
  print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(t_ep_loss, t_ep_acc))

  return t_tags, t_predict, out_t, out_p

tags, predictions, true_seq, pred_seq = test_loop(model_path = './best_model/', test=testing_loader)

print(classification_report(tags, predictions))

"""saving true and predicted labels to new csv"""

true_pred_lables = {"true_labels": true_seq, "predicted_labels": pred_seq}
out_df = pd.DataFrame(true_pred_lables)

out_df.head()

# out_df.to_csv(base_path + "predicted_BERT.csv", index = False)