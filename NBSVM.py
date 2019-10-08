import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files

PATH_TO_IMDB = r'./data/aclImdb'
def load_imdb_data(datadir):
    # read in training and test corpora
    categories = ['pos', 'neg']
    train_b = load_files(datadir + '/train', shuffle=True,
                         categories=categories)
    test_b = load_files(datadir + '/test', shuffle=True,
                        categories=categories)
    train_b.data = [x.decode('utf-8') for x in train_b.data]
    test_b.data = [x.decode('utf-8') for x in test_b.data]
    veczr = CountVectorizer(ngram_range=(1, 3), binary=True,
                            token_pattern=r'\w+',
                            max_features=800000)
    dtm_train = veczr.fit_transform(train_b.data)
    dtm_test = veczr.transform(test_b.data)
    y_train = train_b.target
    y_test = test_b.target
    print("DTM shape (training): (%s, %s)" % (dtm_train.shape))
    print("DTM shape (test): (%s, %s)" % (dtm_train.shape))
    num_words = len([v for k, v in veczr.vocabulary_.items()]) + 1
    print('vocab size:%s' % (num_words))

    return (dtm_train, dtm_test), (y_train, y_test), num_words
(dtm_train, dtm_test), (y_train, y_test), num_words = load_imdb_data(PATH_TO_IMDB)

def dtm2wid(dtm, maxlen):
    x = []
    nwds = []
    for idx, row in enumerate(dtm):
        seq = []
        indices = (row.indices + 1).astype(np.int64)
        np.append(nwds, len(indices))
        data = (row.data).astype(np.int64)
        count_dict = dict(zip(indices, data))
        for k,v in count_dict.items():
            seq.extend([k]*v)
        num_words = len(seq)
        nwds.append(num_words)
        # pad up to maxlen with 0
        if num_words < maxlen:
            seq = np.pad(seq, (maxlen - num_words, 0),
                         mode='constant')
        # truncate down to maxlen
        else:
            seq = seq[-maxlen:]
        x.append(seq)
    nwds = np.array(nwds)
    print('sequence stats: avg:%s, max:%s, min:%s' % (nwds.mean(),
                                                      nwds.max(),
                                                      nwds.min()) )
    return np.array(x)
maxlen = 2000
x_train = dtm2wid(dtm_train, maxlen)
x_test = dtm2wid(dtm_test, maxlen)

def pr(dtm, y, y_i):
    p = dtm[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
nbratios = np.log(pr(dtm_train, y_train, 1)/pr(dtm_train,
                                               y_train, 0))
nbratios = np.squeeze(np.asarray(nbratios))

def get_model(num_words, maxlen, nbratios=None):
    # setup the embedding matrix for NB log-count ratios
    embedding_matrix = np.zeros((num_words, 1))
    for i in range(1, num_words): # skip 0, the padding value
        if nbratios is not None:
            # if log-count ratios are supplied, then it's NBSVM
            embedding_matrix[i] = nbratios[i-1]
        else:
            # if log-count ratios are not supplied,
            # this reduces to a logistic regression
            embedding_matrix[i] = 1    # setup the model
    inp = Input(shape=(maxlen,))
    r = Embedding(num_words, 1, input_length=maxlen,
                  weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Embedding(num_words, 1, input_length=maxlen,
                  embeddings_initializer='glorot_normal')(inp)
    x = dot([r,x], axes=1)
    x = Flatten()(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

model = get_model(num_words, maxlen, nbratios=nbratios)
model.fit(x_train, y_train,
          batch_size=32,
          epochs=3,
          validation_data=(x_test, y_test))
