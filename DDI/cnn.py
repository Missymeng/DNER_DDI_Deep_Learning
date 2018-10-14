from keras import optimizers,callbacks
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Merge, Add
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence

def run_model_fixedembed(dataset,maxlen,embed_dim,hidden_dim,batch_size,filter_sizes,basedir,validate=True,num_epochs=50):

    # Load dataset
    train_lex,valid_lex,test_lex,train_pos,valid_pos,test_pos,train_y,valid_y,test_y = dataset

    # Build CNN model
    print 'Building model...'
    embed_input_shape = (maxlen, embed_dim)
    embed_input = Input(shape=embed_input_shape, name='embed_input')
    z = Dropout(0.25)(embed_input)
    # Convolutional layer: use multiple sizes of filters/kernel sizes to build conv block
    convs_embed = []
    for sz in filter_sizes:
        conv = Convolution1D(batch_size=batch_size,
                             filters=10,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
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
                             filters=3,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z2)
        conv = MaxPooling1D(pool_size=maxlen-sz+1)(conv)
        conv = Flatten()(conv)
        convs_pos.append(conv)

    convs_embed.extend(convs_pos)
    z = Concatenate()(convs_embed)
    z = Dropout(0.25)(z)
    z = Dense(hidden_dim, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)
    model = Model([embed_input,pos_input], model_output)
    print model.summary()

    optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    fileprefix = 'embed_fixed_'

    # Set up callbacks
    am = match.Match(valid_lex,valid_pos,valid_y,pred_dir=os.path.join(basedir,'predictions'),fileprefix=fileprefix)
    mc = callbacks.ModelCheckpoint(os.path.join(basedir,'models','embedfixed.model.weights.{epoch:02d}.hdf5'))
    cbs = [am,mc]
    if validate:
        early_stopping = callbacks.EarlyStopping(monitor='loss',patience=50)
        cbs.append(early_stopping)

    # Train the model
    print 'Training...'
    hist = model.fit([train_lex,train_pos], train_y, batch_size=batch_size, epochs=num_epochs,
                     validation_data=([valid_lex,valid_pos], valid_y),verbose=2,callbacks=cbs)

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
