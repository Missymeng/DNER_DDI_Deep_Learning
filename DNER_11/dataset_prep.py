import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import sys
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import xml.etree.ElementTree as ET
from functions import inti_list, parselist, vectorize
import cPickle as pkl

inputPath = './dataset/DDI2011/all/*.xml'
files = glob.glob(inputPath)

tokset = {'<UNK>'}
tokset |= {'<UNK>'}
tokset |= {'n-[n-(3,','5-difluorophenacetyl)-l-alanyl]-s-phenylglycine','(igf)-1','abeta(1-40)'}
rows = list()
for f in files:
    tree = ET.parse(f)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        row = list()
        id = sentence.get('id')
        text = sentence.get('text')
        tokens = word_tokenize(text)
        row.append(id)
        row.append(text)
        row.append(tokens)

        labelList = inti_list(len(tokens))
        for entity in sentence.findall('entity'):
            name = entity.get('text')
            type = entity.get('type')
            drugWords = word_tokenize(name)  # a drug name can contain several words
            for i in range(0, len(tokens)):
                for k in range(0, len(drugWords)):
                    if k == 0 and tokens[i] == drugWords[k]:
                        labelList[i] = 'B-' + type
                    elif tokens[i] == drugWords[k]:
                        labelList[i] = 'I-' + type

        row.append(labelList)

        lower_text = text.lower()
        lower_tokens = [word.lower() for word in tokens]
        row.append(lower_text)
        row.append(lower_tokens)
        tokset |= set(lower_tokens)
        rows.append(row)

# with open('./dataset/DDI2011_test.csv', "wb") as f:
#     writer = csv.writer(f)
#     writer.writerow(['id','text','tokens','label','lower_text','lower_tokens'])
#     writer.writerows(rows)
# print 'Done'


# Build index dictionaries
labels = ['O','B-drug','I-drug'] + ['<UNK>']
labels2idx = dict(zip(labels, range(1,len(labels)+1)))
tok2idx = dict(zip(tokset, range(1,len(tokset)+1)))  # leave 0 for padding

# Split train/validation from main train set
train_toks_raw = []
train_lex_raw = []
train_y_raw = []
valid_toks_raw = []
valid_lex_raw = []
valid_y_raw = []
t_toks = []
t_lex = []
t_y = []
t_class = []

with open('./dataset/DDI2011_train.csv', 'rb') as f:
    rd = csv.DictReader(f)
    for row in rd:
        t_toks.append(parselist(row['lower_tokens']))
        t_lex.append([row['lower_text']])
        t_y.append(parselist(row['label']))
        t_class.append(row['id'])
        # if '<UNK>' in parselist(row['labels']):
        #     sys.stderr.write('<UNK> found in labels for tweet %s' % row['tokens'])

X_train = []
X_test = []
y_train = []
y_test = []
for i in range(1, 2, 5):
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(t_class, t_y, test_size=0.1, random_state=i)
    X_train.extend(X_train_tmp)
    X_test.extend(X_test_tmp)
    y_train.extend(y_train_tmp)
    y_test.extend(y_test_tmp)
    print len(X_train_tmp)
print len(X_train)

for item in X_train:
    i = t_class.index(item)     # according to value in t_class, get position of the instance in list
    train_toks_raw.append(t_toks[i])
    train_lex_raw.append(t_lex[i])
    train_y_raw.append(t_y[i])
for item in X_test:
    i = t_class.index(item)
    valid_toks_raw.append(t_toks[i])
    valid_lex_raw.append(t_lex[i])
    valid_y_raw.append(t_y[i])

test_toks_raw = []
test_lex_raw = []
test_y_raw = []
with open('./dataset/DDI2011_test.csv', 'rU') as f:
    rd = csv.DictReader(f)
    for row in rd:
        test_toks_raw.append(parselist(row['lower_tokens']))
        test_lex_raw.append([row['lower_text']])
        test_y_raw.append(parselist(row['label']))

# Convert each sentence of normalized tokens and labels into arrays of indices
train_lex = vectorize(train_toks_raw, tok2idx)
train_y = vectorize(train_y_raw, labels2idx)
valid_lex = vectorize(valid_toks_raw, tok2idx)
valid_y = vectorize(valid_y_raw, labels2idx)
test_lex = vectorize(test_toks_raw, tok2idx)
test_y = vectorize(test_y_raw, labels2idx)

# Pickle the resulting data set
with open('./dataset/DDI2011.pkl','w') as fout:
    pkl.dump([[train_toks_raw,train_lex,train_y],[valid_toks_raw,valid_lex,valid_y],[test_toks_raw,test_lex,test_y],
              {'labels2idx':labels2idx, 'words2idx':tok2idx}], fout)
