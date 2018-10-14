import random
import numpy as np
import cPickle as pkl
import sys
import os, shutil
import optparse
import approximateMatch
from matplotlib import pyplot as plt
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from keras.models import Graph, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM
from keras import callbacks
from keras.preprocessing import sequence

def vectorize_set(lexlists, maxlen, V):
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, V])
    for i, lex in enumerate(lexlists):
        for j, tok in enumerate(lex):
            X[i,j,tok] = 1
    return X

def predict_score(model, x, toks, y, pred_dir, i2l, padlen, metafile=None, fileprefix=''):

    ## GRAPH (BIDIRECTIONAL)
    pred_probs = model.predict({'input': x}, verbose=0)['output']
    test_loss = model.evaluate({'input': x, 'output': y}, batch_size=1, verbose=0)
    pred = np.argmax(pred_probs, axis=2)

    N = len(toks)

    # data = pd.read_csv('brands.csv')
    # brands = data['brand'].values

    # If the name of a metafile is passed, simply write this round of predictions to file
    if metafile > 0:
        meta = open(metafile, 'a')

    fname = os.path.join(pred_dir, fileprefix+'approxmatch_test')
    with open(fname, 'w') as fout:
        for i in range(N):
            bos = 'BOS\tO\tO\n'
            fout.write(bos)
            if metafile > 0:
                meta.write(bos)

            sentlen = len(toks[i])
            startind = padlen - sentlen

            preds = [i2l[j] for j in pred[i][startind:]]
            # if y[i] in brands:
            # 	preds[i] = 'B-brand'
            actuals = [i2l[j] for j in np.argmax(y[i], axis=1)[startind:]]
            for (w, act, p) in zip(toks[i], actuals, preds):
                line = '\t'.join([w, act, p])+'\n'
                fout.write(line)
                if metafile > 0:
                    meta.write(line)

            eos = 'EOS\tO\tO\n'
            fout.write(eos)
            if metafile > 0:
                meta.write(eos)
    scores = approximateMatch.get_approx_match(fname)
    scores['loss'] = test_loss
    if metafile > 0:
        meta.close()

    with open(fname, 'a') as fout:
        fout.write('\nTEST Approximate Matching Results:\n  Precision '+ str(scores['p'])+ ' Recall ' + str(scores['r']) + ' F1 ' + str(scores['f1']))
    return scores


if __name__=="__main__":
    pred_dir = './test/predictions'

    # # Load the data
    with open('./dataset/DDI13_processed_10fold.pkl', 'rb') as f:
        train_set, valid_set, test_set, dic = pkl.load(f)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in labels2idx: data may be lost because 0 used as padding\n')
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in words2idx: data may be lost because 0 used as padding\n')
    idx2word[0] = 'PAD'
    idx2label[0] = 'PAD'

    # train_toks, train_lex, train_y = train_set
    # valid_toks, valid_lex, valid_y = valid_set
    test_toks, test_lex,  test_y = test_set

    # validsize = len(valid_toks)
    # maxtrain = int(len(train_toks+valid_toks) * 0.95)
    #
    # trainval_toks = train_toks+valid_toks
    # trainval_lex = train_lex+valid_lex
    # trainval_y = train_y+valid_y
    # dat = zip(trainval_toks, trainval_lex, trainval_y)
    # random.shuffle(dat)
    # trainval_toks, trainval_lex, trainval_y = zip(*dat)
    # train_toks = trainval_toks[:maxtrain]
    # train_lex = trainval_lex[:maxtrain]
    # train_y = trainval_y[:maxtrain]
    # valid_toks = trainval_toks[-validsize:]
    # valid_lex = trainval_lex[-validsize:]
    # valid_y = trainval_y[-validsize:]

    # vocsize =  max(idx2word.keys()) + 1
    nclasses = max(idx2label.keys()) + 1

    # if len(valid_lex) > 0:
    #     validate = True
    # else:
    #     validate = False
    #
    # maxlen = max([len(l) for l in train_lex])
    # if len(valid_lex) > 0:
    #     maxlen = max(maxlen, max([len(l) for l in valid_lex]))
    maxlen = max([len(l) for l in test_lex])

    # Pad inputs to max sequence length and turn into one-hot vectors
    # train_lex = sequence.pad_sequences(train_lex, maxlen=maxlen)
    # valid_lex = sequence.pad_sequences(valid_lex, maxlen=maxlen)
    test_lex = sequence.pad_sequences(test_lex, maxlen=maxlen)

    # train_y = sequence.pad_sequences(train_y, maxlen=maxlen)
    # valid_y = sequence.pad_sequences(valid_y, maxlen=maxlen)
    test_y = sequence.pad_sequences(test_y, maxlen=maxlen)

    # train_y = vectorize_set(train_y, maxlen, nclasses)
    # valid_y = vectorize_set(valid_y, maxlen, nclasses)
    test_y = vectorize_set(test_y, maxlen, nclasses)

    fileprefix = 'embed_fixed_'
    basedir = './test'

    model = Graph()
    bestmodelfile = os.path.join(basedir, 'models','embedfixed.model.weights.02.hdf5')
    model = model_from_json(open(os.path.join(basedir,'models','embedfixed_model_architecture.json')).read())
    model.load_weights(bestmodelfile)

    scores = predict_score(model, test_lex, test_toks, test_y, os.path.join(basedir,'predictions'), idx2label,
                           maxlen, fileprefix=fileprefix)
