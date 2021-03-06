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

def embed_set(lexlists, toklists, maxlen, w2vModel):
    def pad_toks(toklist, padlen):
        padded = ['PAD'] * (padlen-len(toklist))
        padded += toklist
        return np.array(padded)

    ## TODO: Normalize URLs, digits, etc

    dim = w2vModel.wv.syn0.shape[1]
    # dim = w2v.vector_size
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, dim])
    for i, (lex, toklist) in enumerate(zip(lexlists,toklists)):
        toklist = pad_toks(toklist,maxlen)
        for j, tok in enumerate(toklist):
            if tok != 'PAD':
                idx = w2vModel.wv.vocab.get(tok, w2vModel.wv.vocab['the']).index
                vec = w2vModel.wv.syn0[idx]
                X[i,j] = vec
    return X

def learning_curve(history, pltname='history.pdf', preddir=None, fileprefix=''):
    '''
    Plot Validation accuracy/loss and optionally Approximate Match F1
    for each epoch
    :param history: keras.callbacks.History object
    :param preddir: directory containing approximate match results for ApproximateMatch callback
    :return:
    '''
    num_epochs = len(history.history['val_loss'])
    n = range(num_epochs)

    approxmatch = []
    for i in n:
        f1 = open(os.path.join(preddir, fileprefix+'approxmatch_epoch'+str(i)), 'rU').readlines()[-1].strip().split()[-1]
        approxmatch.append(float(f1))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n, history.history['loss'], '-b', label='Trn Loss')
    #ax.plot(n, history.history['acc'], '--b', label='Trn Acc')
    ax.plot(n, history.history['val_loss'], '-r', label='Val Loss')
    #ax.plot(n, history.history['val_acc'], '--r', label='Val Acc')
    ax.plot(n, approxmatch, '-g', label='StrictMatch F1')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.savefig(pltname)
    plt.close()
    sys.stderr.write('Max approximate match F1: %0.2f\n' % max(approxmatch))
    return np.max(approxmatch), np.argmax(approxmatch)


def predict_score(model, x, toks, y, pred_dir, i2l, padlen, metafile=None, fileprefix=''):

    ## GRAPH (BIDIRECTIONAL)
    pred_probs = model.predict({'input': x}, verbose=0)['output']
    test_loss = model.evaluate({'input': x, 'output': y}, batch_size=1, verbose=0)
    pred = np.argmax(pred_probs, axis=2)

    N = len(toks)

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

def run_model_fixedembed(dataset, numhidden, hiddendim, idx2word, idx2label, w2v, basedir, validate=True, num_epochs=30):
    train_toks, valid_toks, test_toks, \
    train_lex, valid_lex, test_lex, \
    train_y, valid_y, test_y = dataset

    maxlen = max([len(l) for l in train_lex])
    if len(valid_lex) > 0:
        maxlen = max(maxlen, max([len(l) for l in valid_lex]))
    maxlen = max(maxlen, max([len(l) for l in test_lex]))

    nclasses = max(idx2label.keys()) + 1
    embed_dim = w2v.wv.syn0.shape[1]
    # embed_dim = w2v.vector_size

    # Pad inputs to max sequence length and turn into one-hot vectors
    train_lex = sequence.pad_sequences(train_lex, maxlen=maxlen)
    valid_lex = sequence.pad_sequences(valid_lex, maxlen=maxlen)
    test_lex = sequence.pad_sequences(test_lex, maxlen=maxlen)

    train_lex = embed_set(train_lex, train_toks, maxlen, w2v)
    valid_lex = embed_set(valid_lex, valid_toks, maxlen, w2v)
    test_lex = embed_set(test_lex, test_toks, maxlen, w2v)

    train_y = sequence.pad_sequences(train_y, maxlen=maxlen)
    valid_y = sequence.pad_sequences(valid_y, maxlen=maxlen)
    test_y = sequence.pad_sequences(test_y, maxlen=maxlen)

    train_y = vectorize_set(train_y, maxlen, nclasses)
    valid_y = vectorize_set(valid_y, maxlen, nclasses)
    test_y = vectorize_set(test_y, maxlen, nclasses)

    # Build the model
    ## BI-DIRECTIONAL
    print('Building the model...')
    H = numhidden
    model = Graph()
    model.add_input(name='input', input_shape=(maxlen, embed_dim))
    model.add_node(Dropout(0.), name='dropout_embedding', input='embedding')
    model.add_node(Masking(), name='mask', input='input')

    # Build first hidden layer
    model.add_node(LSTM(hiddendim, return_sequences=True, activation='tanh'), name='forward0', input='mask')
    model.add_node(Dropout(0.1), name='dropout0f', input='forward0')
    model.add_node(LSTM(hiddendim, return_sequences=True, go_backwards=True, activation='tanh'), name='backwards0', input='mask')
    model.add_node(Dropout(0.1), name='dropout0b', input='backwards0')

    # Build subsequent hidden layers
    if H > 1:
        for i in range(1, H):
            model.add_node(LSTM(hiddendim, return_sequences=True, activation='tanh'), name='forward%d'%i, input='dropout%df'%(i-1))
            model.add_node(Dropout(0.1), name='dropout%df'%i, input='forward%d'%i)
            model.add_node(LSTM(hiddendim, return_sequences=True, go_backwards=True, activation='tanh'), name='backwards%d'%i, input='dropout%db'%(i-1))
            model.add_node(Dropout(0.1), name='dropout%db'%i, input='backwards%d'%i)

    # Finish up the network
    model.add_node(TimeDistributedDense(nclasses), name='tdd', inputs=['dropout%df'%(H-1),'dropout%db'%(H-1)], merge_mode='ave')
    model.add_node(Activation('softmax'), name='softmax', input='tdd')
    model.add_output(name='output', input='softmax')
    model.compile(optimizer='rmsprop', loss={'output': 'categorical_crossentropy'})

    # Set up callbacks
    fileprefix = 'embed_fixed_'
    am = approximateMatch.ApproximateMatch_SEQ(valid_toks, valid_y, valid_lex, idx2label,
                                        pred_dir=os.path.join(basedir, 'predictions'), fileprefix=fileprefix)
    mc = callbacks.ModelCheckpoint(os.path.join(basedir, 'models', 'embedfixed.model.weights.{epoch:02d}.hdf5'))
    cbs = [am, mc]
    if validate:
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        cbs.append(early_stopping)

    # Train the model
    print('Training...')
    hist = model.fit({'input': train_lex, 'output': train_y}, nb_epoch=num_epochs, batch_size=1,
                     validation_data={'input': valid_lex, 'output': valid_y},
                     callbacks=cbs)
    if validate:
        val_f1, best_model = learning_curve(hist, preddir=os.path.join(basedir, 'predictions'),
                                            pltname=os.path.join(basedir, 'charts', 'hist_fixedembed%d_nhidden%d.pdf'
                                                    % (hiddendim, numhidden)), fileprefix=fileprefix)
    else:
        best_model = num_epochs-1
        val_f1 = 0.0

    # Save model
    json_string = model.to_json()
    open(os.path.join(basedir, 'models', 'embedfixed_model_architecture.json'),'w').write(json_string)

    # Test
    bestmodelfile = os.path.join(basedir, 'models','embedfixed.model.weights.%02d.hdf5' % best_model)
    shutil.copyfile(bestmodelfile, bestmodelfile.replace('.hdf5','.best.hdf5'))
    if validate:
        model = model_from_json(open(os.path.join(basedir,'models','embedfixed_model_architecture.json')).read())
        model.load_weights(bestmodelfile)

    scores = predict_score(model, test_lex, test_toks, test_y, os.path.join(basedir, 'predictions'), idx2label,
                            maxlen, fileprefix=fileprefix)

    scores['val_f1'] = val_f1
    return scores, hist.history, best_model

def init_embedding_weights(i2w, w2vmodel):
    # Create initial embedding weights matrix
    # Return: np.array with dim [vocabsize, embeddingsize]

    d = w2vmodel.wv.syn0.shape[1]
    V = len(i2w)
    assert sorted(i2w.keys()) == range(V)  # verify indices are sequential

    emb = np.zeros([V,d])
    for i,l in i2w.items():
        if i==0:
            continue
        if l in w2vmodel.wv.vocab:
            w2v_index_i = w2vmodel.wv.vocab.get(l, w2vmodel.wv.vocab['the']).index
            emb[i] = w2vmodel.wv.syn0[w2v_index_i]
        else:
            emb[i] = np.random.uniform(-1.0, 1.0, d)
    return emb


def run_model_varyembed(dataset, numhidden, hiddendim, idx2word, idx2label, w2v, basedir,
                        embedding_dim=400, validate=True, num_epochs=30):

    train_toks, valid_toks, test_toks, \
    train_lex, valid_lex, test_lex, \
    train_y, valid_y, test_y = dataset

    maxlen = max([len(l) for l in train_lex])
    if len(valid_lex) > 0:
        maxlen = max(maxlen, max([len(l) for l in valid_lex]))
    maxlen = max(maxlen, max([len(l) for l in test_lex]))

    vocsize = max(idx2word.keys()) + 1
    nclasses = max(idx2label.keys()) + 1

    # Pad inputs to max sequence length and turn into one-hot vectors
    train_lex = sequence.pad_sequences(train_lex, maxlen=maxlen)
    valid_lex = sequence.pad_sequences(valid_lex, maxlen=maxlen)
    test_lex = sequence.pad_sequences(test_lex, maxlen=maxlen)

    train_y = sequence.pad_sequences(train_y, maxlen=maxlen)
    valid_y = sequence.pad_sequences(valid_y, maxlen=maxlen)
    test_y = sequence.pad_sequences(test_y, maxlen=maxlen)

    train_y = vectorize_set(train_y, maxlen, nclasses)
    valid_y = vectorize_set(valid_y, maxlen, nclasses)
    test_y = vectorize_set(test_y, maxlen, nclasses)


    # Build the model
    ## BI-DIRECTIONAL
    print('Building the model...')
    H = numhidden
    model = Graph()
    model.add_input(name='input', input_shape=[maxlen], dtype='int')

    # Add embedding layer
    if w2v is None:
        model.add_node(Embedding(vocsize, embedding_dim, init='lecun_uniform', input_length=maxlen), name='embed', input='input')
    else:
        embeds = init_embedding_weights(idx2word, w2v)
        embed_dim = w2v.wv.syn0.shape[1]
        model.add_node(Embedding(vocsize, embed_dim, input_length=maxlen, weights=[embeds], mask_zero=True), name='embed', input='input')

    # Build first hidden layer
    model.add_node(LSTM(hiddendim, return_sequences=True, activation='tanh'), name='forward0', input='embed')
    model.add_node(Dropout(0.1), name='dropout0f', input='forward0')
    model.add_node(LSTM(hiddendim, return_sequences=True, go_backwards=True, activation='tanh'), name='backwards0', input='embed')
    model.add_node(Dropout(0.1), name='dropout0b', input='backwards0')

    # Build subsequent hidden layers
    if H > 1:
        for i in range(1, H):
            model.add_node(LSTM(hiddendim, return_sequences=True, activation='tanh'), name='forward%d'%i, input='dropout%df'%(i-1))
            model.add_node(Dropout(0.1), name='dropout%df'%i, input='forward%d'%i)
            model.add_node(LSTM(hiddendim, return_sequences=True, go_backwards=True, activation='tanh'), name='backwards%d'%i, input='dropout%db'%(i-1))
            model.add_node(Dropout(0.1), name='dropout%db'%i, input='backwards%d'%i)


    # Finish up the network
    model.add_node(TimeDistributedDense(nclasses), name='tdd', inputs=['dropout%df'%(H-1),'dropout%db'%(H-1)], merge_mode='ave')
    model.add_node(Activation('softmax'), name='softmax', input='tdd')
    model.add_output(name='output', input='softmax')
    model.compile(optimizer='rmsprop', loss={'output': 'categorical_crossentropy'})


    # Set up callbacks
    fileprefix = 'embed_varied_'
    am = approximateMatch.ApproximateMatch_SEQ(valid_toks, valid_y, valid_lex, idx2label,
                                        pred_dir=os.path.join(basedir, 'predictions'), fileprefix=fileprefix)
    mc = callbacks.ModelCheckpoint(os.path.join(basedir, 'models', 'embedding.model.weights.{epoch:02d}.hdf5'))
    cbs = [am, mc]
    if validate:
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
        cbs.append(early_stopping)

    # Train the model
    print('Training...')
    hist = model.fit({'input': train_lex, 'output': train_y}, nb_epoch=num_epochs, batch_size=1,
                     validation_data={'input': valid_lex, 'output': valid_y},
                     callbacks=cbs)
    if validate:
        val_f1, best_model = learning_curve(hist, preddir=os.path.join(basedir,'predictions'),
                                            pltname=os.path.join(basedir,'charts','hist_varyembed%d_nhidden%d.pdf'
                                                    % (hiddendim, numhidden)), fileprefix=fileprefix)
    else:
        best_model = num_epochs-1
        val_f1 = 0.0

    # Save model
    json_string = model.to_json()
    open(os.path.join(basedir,'models','embedding_model_architecture.json'),'w').write(json_string)

    # Test
    bestmodelfile = os.path.join(basedir, 'models','embedding.model.weights.%02d.hdf5' % best_model)
    shutil.copyfile(bestmodelfile, bestmodelfile.replace('.hdf5','.best.hdf5'))
    if validate:
        model = model_from_json(open(os.path.join(basedir,'models','embedding_model_architecture.json')).read())
        model.load_weights(bestmodelfile)


    scores = predict_score(model, test_lex, test_toks, test_y, os.path.join(basedir,'predictions'), idx2label,
                           maxlen, fileprefix=fileprefix)

    scores['val_f1'] = val_f1

    return scores, hist.history, best_model


if __name__=="__main__":

    # # Load the data
    with open('./dataset/DDI13_processed_10fold_testOnDrugbank.pkl', 'rb') as f:
        train_set, valid_set, test_set, dic = pkl.load(f)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in labels2idx: data may be lost because 0 used as padding\n')
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in words2idx: data may be lost because 0 used as padding\n')
    idx2word[0] = 'PAD'
    idx2label[0] = 'PAD'


    train_toks, train_lex, train_y = train_set
    valid_toks, valid_lex, valid_y = valid_set
    test_toks, test_lex,  test_y = test_set

    validsize = len(valid_toks)
    maxtrain = int(len(train_toks+valid_toks) * 0.95)

    trainval_toks = train_toks+valid_toks
    trainval_lex = train_lex+valid_lex
    trainval_y = train_y+valid_y
    dat = zip(trainval_toks, trainval_lex, trainval_y)
    random.shuffle(dat)
    trainval_toks, trainval_lex, trainval_y = zip(*dat)
    train_toks = trainval_toks[:maxtrain]
    train_lex = trainval_lex[:maxtrain]
    train_y = trainval_y[:maxtrain]
    valid_toks = trainval_toks[-validsize:]
    valid_lex = trainval_lex[-validsize:]
    valid_y = trainval_y[-validsize:]

    vocsize =  max(idx2word.keys()) + 1
    nclasses = max(idx2label.keys()) + 1

    dataset = (train_toks, valid_toks, test_toks, train_lex, valid_lex, test_lex, train_y, valid_y, test_y)
    if len(valid_lex) > 0:
        validate = True
    else:
        validate = False

    # load pretrained embedding model
    w2v = word2vec.Word2Vec.load_word2vec_format('./model/prevocab_dim_440.txt', binary=False)
    embed_dim = w2v.vector_size

    varyembeddings = 'False'
    randinit = 'False'
    numhidden = 1
    hiddendim = 256
    basedir = './model_output'
    nbepochs = 1
    # scores, history, best_model = run_model_fixedembed(dataset, numhidden, hiddendim, idx2word,
    #                                                     idx2label, w2v, basedir, validate=validate,
    #                                                     num_epochs=nbepochs)

    scores, history, best_model = run_model_varyembed(dataset, numhidden, hiddendim, idx2word,
                                                      idx2label, w2v, basedir, validate=validate,
                                                      num_epochs=nbepochs)
    # Retrieve scores
    if validate:
        val_loss = history['val_loss'][best_model]
        val_f1 = scores['val_f1']

    training_loss = history['loss'][best_model]
    test_f1 = scores['f1']
    test_loss = scores['loss']
    print('Scores for # hidden layers: %d\n'
          'hidden dim %d\n'
          'embeddings varied: %s\n'
          'randomly initialized embeddings: %s\n'
          'training set size: %d\n'
          '--------------------------\n'
          'training loss %0.4f\n'
          % (numhidden, hiddendim,
             varyembeddings, randinit,
             len(train_lex), training_loss))
    if validate:
        print('validation loss %0.4f\n'
              'validation f1 %0.4f\n'
              % (val_loss, val_f1))
    print('test loss %0.4f\n'
          'test f1 %0.4f\n'
          % (test_loss, test_f1))


