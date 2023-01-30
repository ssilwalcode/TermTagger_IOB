""" Preparing/Preprocessing data"""
import pandas as pd
import re
import os
import glob
import csv


def dataframe(filepath=""):
    l = []
    for file in glob.glob(os.path.join(filepath, "*.final")):
        with open(file, 'r') as f:
            texts = [tuple(text.strip().split()) for text in f if len(text.strip().split()) == 2]
            j = 0
            for i, (a,b) in enumerate(texts):
                if a == ".":
                    t = ' '.join(a for a,_ in texts[j:i+1])
                    k = ' '.join(b for _,b in texts[j:i+1])
                    j = i+1
                    l.append([t,k])
    return pd.DataFrame(l, columns=['sent', 'tags'])


train_data = dataframe('train')
test_data = dataframe('test')
dev_data = dataframe('dev')

# train_data.to_csv('train.csv', index=False)
# dev_data.to_csv('dev.csv', index=False)
# test_data.to_csv('test.csv', index=False)