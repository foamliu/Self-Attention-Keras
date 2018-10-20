from __future__ import print_function

from keras.datasets import reuters
from keras.preprocessing import sequence

from attention import Position_Embedding, Attention

max_features = 20000
maxlen = 80
batch_size = 32
num_classes = 46

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(num_classes, activation='softmax')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
print(model.summary())
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
