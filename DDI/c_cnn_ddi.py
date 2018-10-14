import numpy as np
import random
import cPickle as pkl
import sys
import os,shutil
import c_match
from matplotlib import pyplot as plt
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import optimizers,callbacks,regularizers, utils
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Merge, Add
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
np.random.seed(0)

def y_reshape(lists):
    ''' reshape y to fit in CNN input
    input format: [[1][2]...[1][2]]
    return format: numpy array [0 1 ... 0 1]
    '''
    y = []
    for list in lists:
        y.append(int(list[0])-1)
    y = np.array(y)
    return y

def pos_reshape(pair_labels,dis1,dis2,maxlen):
    ''' reshape position value to fit in CNN input
    :param pair_labels: list of drug pair labels,[num_sample,maxlen]
    :param dis1: list of distances to drug1, [num_sample,maxlen]
    :param dis2: list of distances to drug2, [num_sample,maxlen]
    :param maxlen: max length of sentence
    :return: numpy array[numsample,maxlen,3]
    '''
    num_sample = len(pair_labels)
    pair_labels = sequence.pad_sequences(pair_labels,maxlen=maxlen)
    dis1 = sequence.pad_sequences(dis1,maxlen=maxlen,value=2000)
    dis2 = sequence.pad_sequences(dis2,maxlen=maxlen,value=2000)

    y = []
    for i in range(0, num_sample):
        y_sample = []
        for j in range(0, maxlen):
            y_item = []
            y_item.append(pair_labels[i][j])
            # y_item.append(dis1[i][j])
            # y_item.append(dis2[i][j])
            y_item.append(np.float32(dis1[i][j])/maxlen)
            y_item.append(np.float32(dis2[i][j])/maxlen)
            y_sample.append(y_item)
        y.append(y_sample)
    y = np.array(y)
    y = np.float32(y)
    return y

def embed_set(lexlists, toklists, maxlen, w2vModel):
    '''generate a numpy array [nb_sample,maxlen,embeddim] to fit in CNN input
    param lexlists: [[word_id,..,word_id][]...[]]
    param toklists: [[word,..,word][]...[]]
    param maxlen: max len of sentences
    return: [nb_sample,maxlen,embeddim]
    '''
    def pad_toks(toklist, padlen):
        padded = ['PAD'] * (padlen-len(toklist))
        padded += toklist
        return np.array(padded)

    dim = w2vModel.vector_size
    nb_samples = len(lexlists)
    X = np.zeros([nb_samples, maxlen, dim])
    for i, (lex, toklist) in enumerate(zip(lexlists,toklists)):
        toklist = pad_toks(toklist,maxlen)
        for j, tok in enumerate(toklist):
            if tok != 'PAD':
                idx = w2vModel.wv.vocab.get(tok, w2vModel.wv.vocab['the']).index
                vec = w2vModel.wv.syn0[idx]
                X[i,j] = vec
    X = np.float32(X)
    return X

def learning_curve(history,pltname='history.pdf',preddir=None,fileprefix=''):
    '''Plot loss,valid loss, and optionally Match F1 for each epoch
    param history: keras.callbacks.History object
    param preddir: directory of prediction results and match scores for Match callback
    return: highest F1 score and at which epoch achieving it
    '''
    num_epoch = len(history.history['val_loss'])
    n = range(num_epoch)

    match = []
    for i in n:
        # read F1 value in 'predictions' files
        f1 = open(os.path.join(preddir,fileprefix+'match_epoch'+str(i)),'rU').readlines()[-1].strip().split()[-1]
        match.append(float(f1))

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(n,history.history['loss'],'-b',label='Trn Loss')
    ax.plot(n,history.history['val_loss'],'-r',label='Val Loss')
    ax.plot(n,match,'-g',label='F1 score')

    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.savefig(pltname)
    plt.close()
    sys.stderr.write('Max matching F1: %0.2f\n' % max(match))

    return np.max(match),np.argmax(match)

def predict_score(model,test_lex,test_pos,test_y,optimizer,pred_dir,metafile=None, fileprefix=''):
    '''given model, predict y for test_x and evaluate'''
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    pred_probs = model.predict([test_lex,test_pos],batch_size=30)
    test_loss = model.evaluate([test_lex,test_pos],test_y,verbose=0)
    print 'Test Loss: ',test_loss

    pred = np.argmax(pred_probs, axis=1)

    # If the name of a metafile is passed, simply write this round of predictions to file
    if metafile > 0:
        meta = open(metafile, 'a')

    y_list = []
    for item in test_y:
        y_list.append(np.where(item==1)[0][0])

    fname = os.path.join(pred_dir,fileprefix+'match_test')
    with open(fname,'w') as fout:
        for i in range(len(pred)):
            fout.write(str(y_list[i])+'\t'+str(pred[i])+'\n')

    scores = c_match.get_match(fname)
    scores['loss'] = test_loss

    with open(fname,'a') as fout:
        fout.write('\nTest Matching Results:\n Precision '+ str(scores['p'])+ ' Recall ' + str(scores['r']) + ' F1 ' + str(scores['f1']))

    return scores

def run_model_fixedembed(dataset,maxlen,embed_dim,hidden_dim,batch_size,filter_sizes,basedir,validate=True,num_epochs=50):

    # Load dataset
    train_lex,valid_lex,test_lex,train_pos,valid_pos,test_pos,train_y,valid_y,test_y = dataset

    # Build CNN model
    print 'Building model...'
    embed_input_shape = (maxlen, embed_dim)
    embed_input = Input(shape=embed_input_shape, name='embed_input')
    z = Dropout(0.5)(embed_input)
    # Convolutional layer: use multiple sizes of filters/kernel sizes to build conv block
    convs_embed = []
    for sz in filter_sizes:
        conv = Convolution1D(batch_size=batch_size,
                             filters=20,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1,
                             bias_regularizer=regularizers.l2(0.1))(z)
        conv = MaxPooling1D(pool_size=maxlen-sz+1)(conv)
        conv = Flatten()(conv)
        convs_embed.append(conv)

    pos_input_shape = (maxlen,3)
    pos_input = Input(shape=pos_input_shape, name='pos_input')
    z2 = Dropout(0.25)(pos_input)
    # convolutional layer: for position columns
    convs_pos = []
    for sz in filter_sizes:
        conv = Convolution1D(batch_size=batch_size,
                             filters=10,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1,
                             bias_regularizer=regularizers.l2(0.1))(z2)
        conv = MaxPooling1D(pool_size=maxlen-sz+1)(conv)
        conv = Flatten()(conv)
        convs_pos.append(conv)

    convs_embed.extend(convs_pos)
    z = Concatenate()(convs_embed)
    z = Dropout(0.5)(z)
    z = Dense(hidden_dim, activation="relu",bias_regularizer=regularizers.l2(0.1))(z)
    model_output = Dense(5, activation="softmax")(z)
    model = Model([embed_input,pos_input], model_output)
    print model.summary()

    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    fileprefix = 'embed_fixed_'

    # Set up callbacks
    am = c_match.StrictMatch(valid_lex,valid_pos,valid_y,pred_dir=os.path.join(basedir,'predictions'),fileprefix=fileprefix)
    mc = callbacks.ModelCheckpoint(os.path.join(basedir,'models','embedfixed.model.weights.{epoch:02d}.hdf5'))
    cbs = [am,mc]
    if validate:
        early_stopping = callbacks.EarlyStopping(monitor='loss',patience=6)
        cbs.append(early_stopping)

    # Train the model
    print 'Training...'
    hist = model.fit([train_lex,train_pos], train_y, batch_size=batch_size, epochs=num_epochs,
                     validation_data=([valid_lex,valid_pos], valid_y),verbose=2,callbacks=cbs)

    # pred_probs = model.predict([test_lex,test_pos],batch_size=1)
    # print pred_probs

    # pred = np.argmax(pred_probs, axis=1)
    # print pred

    if validate:
        val_f1, best_model = learning_curve(hist, preddir=os.path.join(basedir,'predictions'),
                                        pltname=os.path.join(basedir,'charts','hist_fixedembed_hiddendim%d.pdf' % hidden_dim),
                                        fileprefix=fileprefix)
    else:
        best_model = num_epochs-1
        val_f1 = 0.0

    # Save model
    json_string = model.to_json()
    open(os.path.join(basedir,'models','embedfixed_model_architecture.json'),'w').write(json_string)

    # Test
    best_model_file = os.path.join(basedir,'models','embedfixed.model.weights.%02d.hdf5' % best_model)
    shutil.copyfile(best_model_file,best_model_file.replace('.hdf5','.best.hdf5'))

    if validate:
        model = model_from_json(open(os.path.join(basedir,'models','embedfixed_model_architecture.json')).read())
        model.load_weights(best_model_file)

    scores = predict_score(model,test_lex,test_pos,test_y,optimizer,os.path.join(basedir,'predictions'),fileprefix=fileprefix)
    scores['val_f1'] = val_f1

    return scores, hist.history, best_model

def data_loader(datafile,w2vfile):

    with open(datafile,'rb') as f:
        train_set, valid_set, test_set, dic = pkl.load(f)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2pairLabel = dict((k,v) for v,k in dic['pairLabels2idx'].iteritems())
    idx2word = dict((k,v) for v,k in dic['words2idx'].iteritems())
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in labels2idx: data may be lost because 0 used as padding\n')
    if 0 in idx2word:
        sys.stderr.write('Index 0 found in words2idx: data may be lost because 0 used as padding\n')
    idx2word[0] = 'PAD'
    idx2label[0] = 'PAD'
    print 'ddi label dictionary:', idx2label

    train_toks, train_lex, train_pair_labels, train_dis1, train_dis2, train_y = train_set
    valid_toks, valid_lex, valid_pair_labels, valid_dis1, valid_dis2, valid_y = valid_set
    test_toks, test_lex, test_pair_labels, test_dis1, test_dis2, test_y = test_set

    ## # Preprocess loaded data to fit in CNN model
    maxlen = max([len(l) for l in train_lex])
    if len(valid_lex) > 0:
        maxlen = max(maxlen, max([len(l) for l in valid_lex]))
    maxlen = max(maxlen, max([len(l) for l in test_lex]))
    print 'max len of sentence:', maxlen

    nclasses = max(idx2label.keys()) + 1

    # pad inputs to max sequence length and turn into one-hot vectors
    train_lex = sequence.pad_sequences(train_lex, maxlen=maxlen)
    valid_lex = sequence.pad_sequences(valid_lex, maxlen=maxlen)
    test_lex = sequence.pad_sequences(test_lex, maxlen=maxlen)
    print 'train lex shape:',train_lex.shape

    # load pre-trained embedding model
    w2v = word2vec.Word2Vec.load_word2vec_format(w2vfile, binary=False)
    embed_dim = w2v.vector_size
    # embed_dim = w2v.wv.syn0.shape[1]

    # shape sentences of tokens to np array of [num_sen,maxlen,embed_dim]
    train_lex = embed_set(train_lex, train_toks, maxlen, w2v)
    valid_lex = embed_set(valid_lex, valid_toks, maxlen, w2v)
    test_lex = embed_set(test_lex, test_toks, maxlen, w2v)
    print 'train lex shape after embed_set:',train_lex.shape

    # shape y for CNN binary classification
    train_y = y_reshape(train_y)
    valid_y = y_reshape(valid_y)
    test_y = y_reshape(test_y)
    print 'test y shape after reshape', test_y.shape
    print 'train y', train_y
    print 'test y', test_y
    train_y = utils.to_categorical(train_y,num_classes=5)
    valid_y = utils.to_categorical(valid_y,num_classes=5)
    test_y = utils.to_categorical(test_y,num_classes=5)
    print 'test y shape after to_categorical', test_y.shape

    # reshape cols of train_pair_label,train_dis1,train_dis2
    train_pos = pos_reshape(train_pair_labels,train_dis1,train_dis2,maxlen)
    valid_pos = pos_reshape(valid_pair_labels,valid_dis1,valid_dis2,maxlen)
    test_pos = pos_reshape(test_pair_labels,test_dis1,test_dis2,maxlen)
    print 'train pos shape after reshape:',train_pos.shape
    # print train_pos[1]

    dataset = (train_lex,valid_lex,test_lex,train_pos,valid_pos,test_pos,train_y,valid_y,test_y)
    return dataset,maxlen,embed_dim

if __name__=='__main__':

    # y = [1,3,1,3]
    # y = np.array(y)
    # print y
    # y1 = utils.to_categorical(y,num_classes=3)
    # print y1

    datafile = './dataset/dataset_dep/ddic_oversampling_negFiltered_dep.pkl'
    w2vfile = './model/prevocab_dim_440.txt'
    dataset, maxlen, embed_dim = data_loader(datafile,w2vfile)

    basedir = './c_model_output'
    validate = True
    hidden_dim = 256
    batch_size = 50
    filter_sizes = [3,4,5,6,7]
    num_epoches = 18

    scores,history,best_model = run_model_fixedembed(dataset,maxlen,embed_dim,hidden_dim,batch_size,filter_sizes,
                                                 basedir,validate=validate,num_epochs=num_epoches)

    # Retrieve scores
    if validate:
        val_loss = history['val_loss'][best_model]
        val_f1 = scores['val_f1']

    training_loss = history['loss'][best_model]
    test_f1 = scores['f1']
    test_loss = scores['loss']

    print 'Scores for # hidden dim %d \n' % hidden_dim, 'Training loss: %0.4f\n' % training_loss
    if validate:
        print 'Validation loss %0.4f\nValidation F1 %0.4f\n' % (val_loss,val_f1)
    print 'Test loss %0.4f\nTest F1 %0.4f\n' % (test_loss,test_f1)
