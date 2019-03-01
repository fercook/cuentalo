import keras
import spacy
from numpy.random import seed
import os
from numpy import asarray
from pandas import read_pickle
import os
from pandas import DataFrame, concat, read_csv
from numpy import isfinite
from sklearn.preprocessing import LabelEncoder
from numpy import asarray, arange
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Activation,  Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM
from numpy import zeros
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

seed_ = 10
seed(seed_)

glove_dir = '../data'
embeddings_index = {}
with open(os.path.join(glove_dir, 'glove.6B.300d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))


df = read_pickle('../data/dataset_previo_final.pkl')
df = df[df.tweet_type =='original']

concat_df_labels = DataFrame()
cat_path = '../data/Categorizacion/'
for i in os.listdir(cat_path):
    print(cat_path + i)
    df_labels = read_csv(cat_path + i, sep=';', error_bad_lines=False)

    df_labels = df_labels[(df_labels['QUIEN'].astype(str) != '?')]
    df_labels['QUIEN'] = df_labels['QUIEN'].astype(float)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: x - 1)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 0 if x == 0 or x == 1 else x)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 1 if x == 2 else x)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 2 if x == 3 or x == 4 else x)

    df_labels['full_text'] = df_labels['full_text'].astype(str)

    df_labels['Asesinato'] = df_labels['Asesinato'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Violacion'] = df_labels['Violacion'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Agresion \nsexual'] = df_labels['Agresion \nsexual'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Maltrato'] = df_labels['Maltrato'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Acoso'] = df_labels['Acoso'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Miedo'] = df_labels['Miedo'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Asco\nTristeza\nRabia'] = df_labels['Asco\nTristeza\nRabia'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)

    df_labels = df_labels[['id','tweet_id', 'user_name', 'QUIEN', 'Asesinato', 'Violacion', 'Agresion \nsexual', 'Maltrato', 'Acoso', 'Miedo', 'Asco\nTristeza\nRabia', 'full_text']]
    df_labels.set_index('id', inplace=True)
    df_labels = df_labels[isfinite(df_labels['QUIEN'])]

    concat_df_labels = concat([concat_df_labels, df_labels], axis=0)

concat_df_labels.describe()

concat_df_labels.sample(5)

concat_df_labels.columns

def get_category(x):
    if x['Asesinato']:
        return 0
    elif x['Violacion']:
        return 1
    elif x['Agresion \nsexual']:
        return 2
    elif x['Maltrato']:
        return 3
    elif x['Acoso']:
        return 4
    elif x['Miedo']:
        return 5
    elif x['Asco\nTristeza\nRabia']:
        return 6
    else:
        return 7

text_labels = ['Asesinato', 'Violacion', 'Agresion \nsexual', 'Maltrato', 'Acoso', 'Miedo', 'Asco\nTristeza\nRabia']

concat_df_labels['category'] = concat_df_labels[text_labels].apply(get_category, axis=1)
concat_df_labels['unlabeled'] = concat_df_labels['category'].apply(lambda x: 1 if x == 7 else 0)


nlp = spacy.load('es')

def keep_meaningful_words(word):
    processed = nlp(word)
    result = [token.lemma_ for token in processed if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
    return ' '.join(result)

random_tweet = df['full_text'].iloc[1001]
print(random_tweet)
print('-' * 30)
print(keep_meaningful_words(random_tweet))

concat_df_labels['filtered_text'] = concat_df_labels['full_text'].apply(keep_meaningful_words)

concat_df_labels.sample()


enc = LabelEncoder()
y = enc.fit_transform(concat_df_labels['QUIEN'])
concat_df_labels['y'] = y

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

labels = onehot_encoded

n_classes = 3

concat_df_labels.shape

texts = concat_df_labels['full_text'].values.tolist()

len(texts)

input_length = 100
training_samples = 6323
validation_samples = 2000
max_words = 30000
maxlen=100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples+validation_samples:]
y_test = labels[training_samples+validation_samples:]

concat_df_labels.QUIEN.value_counts() / concat_df_labels.QUIEN.value_counts().sum()


labels = asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = arange(labels.shape[0])
data = data[indices]
labels = labels[indices]

concat_df_labels.head()

text_labels = ['1a persona o 2a persona', 'apoyo', 'others & trolls']


embedding_dim = 300
vocabulary_size = 30000
embedding_matrix = zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

NUM_EPOCHS=4
BATCH_SIZE=16

## MLP dumb


seed(seed_)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

y_train.shape

seed(seed_)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# MLP
seed(seed_)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

y_train.shape

seed(seed_)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))


## CNN + LSTM

seed(seed_)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(50))#, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

seed(seed_)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict_classes(x_test)

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))



########################
    # WHAT

import keras
import os
from numpy import asarray
from numpy.random import seed
import os
from pandas import DataFrame, concat, read_csv
from numpy import isfinite
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
from numpy import zeros
from numpy import asarray, arange
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM
from sklearn.metrics import average_precision_score
import numpy as np

seed(seed_)

df = read_pickle('../data/dataset_previo_final.pkl')
df = df[df.tweet_type =='original']


concat_df_labels = DataFrame()
cat_path = '../data/Categorizacion/'
for i in os.listdir(cat_path):
    print(cat_path + i)
    df_labels = read_csv(cat_path + i, sep=';', error_bad_lines=False)

    df_labels = df_labels[(df_labels['QUIEN'].astype(str) != '?')]
    df_labels['QUIEN'] = df_labels['QUIEN'].astype(float)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: x - 1)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 0 if x == 0 or x == 1 else x)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 1 if x == 2 else x)
    df_labels['QUIEN'] = df_labels['QUIEN'].apply(lambda x: 2 if x == 3 or x == 4 else x)

    df_labels['full_text'] = df_labels['full_text'].astype(str)

    df_labels['Asesinato'] = df_labels['Asesinato'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Violacion'] = df_labels['Violacion'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Agresion \nsexual'] = df_labels['Agresion \nsexual'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Maltrato'] = df_labels['Maltrato'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Acoso'] = df_labels['Acoso'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Miedo'] = df_labels['Miedo'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)
    df_labels['Asco\nTristeza\nRabia'] = df_labels['Asco\nTristeza\nRabia'].apply(lambda x: 1 if x == 'x' or x=='X' else 0)

    df_labels = df_labels[['id','tweet_id', 'user_name', 'QUIEN', 'Asesinato', 'Violacion', 'Agresion \nsexual', 'Maltrato', 'Acoso', 'Miedo', 'Asco\nTristeza\nRabia', 'full_text']]
    df_labels.set_index('id', inplace=True)
    df_labels = df_labels[isfinite(df_labels['QUIEN'])]

    concat_df_labels = concat([concat_df_labels, df_labels], axis=0)

concat_df_labels.describe()

concat_df_labels.sample(5)

concat_df_labels.columns

def get_category(x):
    if x['Asesinato']:
        return 0
    elif x['Violacion']:
        return 1
    elif x['Agresion \nsexual']:
        return 2
    elif x['Maltrato']:
        return 3
    elif x['Acoso']:
        return 4
    elif x['Miedo']:
        return 5
    elif x['Asco\nTristeza\nRabia']:
        return 6
    else:
        return 7

text_labels = ['Asesinato', 'Violacion', 'Agresion \nsexual', 'Maltrato', 'Acoso', 'Miedo', 'Asco\nTristeza\nRabia']
concat_df_labels['category'] = concat_df_labels[text_labels].apply(get_category, axis=1)
concat_df_labels['unlabeled'] = concat_df_labels['category'].apply(lambda x: 1 if x == 7 else 0)
concat_df_labels['category'] = concat_df_labels['category'].apply(lambda x: 0 if x == 0 or x == 1 or x == 2 or x == 3 else x)
concat_df_labels['category'] = concat_df_labels['category'].apply(lambda x: 1 if x == 4 or x == 5 or x == 6 else x)
concat_df_labels['category'] = concat_df_labels['category'].apply(lambda x: 2 if x == 7 else x)

nlp = spacy.load('es')

def keep_meaningful_words(word):
    processed = nlp(word)
    result = [token.lemma_ for token in processed if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
    return ' '.join(result)

random_tweet = df['full_text'].iloc[1001]
print(random_tweet)
print('-' * 30)
print(keep_meaningful_words(random_tweet))

concat_df_labels['filtered_text'] = concat_df_labels['full_text'].apply(keep_meaningful_words)

concat_df_labels.sample()

enc = LabelEncoder()
y = enc.fit_transform(concat_df_labels['category'])
concat_df_labels['y'] = y


onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

labels = onehot_encoded

n_classes = 3

texts = concat_df_labels['full_text'].values.tolist()

input_length = 100
training_samples = 6323
validation_samples = 2000
max_words = 30000
maxlen = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples+validation_samples:]
y_test = labels[training_samples+validation_samples:]

concat_df_labels.QUIEN.value_counts() / concat_df_labels.QUIEN.value_counts().sum()


labels = asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = arange(labels.shape[0])
data = data[indices]
labels = labels[indices]

concat_df_labels.head()

text_labels = ['1a persona o 2a persona', 'apoyo', 'others & trolls']


embedding_dim = 300
vocabulary_size = 30000
embedding_matrix = zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

NUM_EPOCHS=4
BATCH_SIZE=16

## MLP dumb

seed(seed_)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

y_train.shape

seed(seed_)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

## MLP

seed(seed_)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

y_train.shape

seed(seed_)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

y_test.argmax(axis=1)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
## CNN + LSTM

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(50))#, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.hy')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', alpha=0.8)
plt.plot(epochs, val_loss, 'g', alpha=0.8)
plt.title('Training and validation loss')
plt.show()

model.metrics_names

score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE, verbose=1)

print('Test accuracy:', score[1])

y_pred = model.predict_classes(x_test)

y_pred = model.predict(x_test)
preds = y_pred.argmax(axis=1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = preds.reshape(len(preds), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))#, labels=text_labels)
print('*'*40)
print(cm)
print('*'*40)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
