import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
]

labels = np.array([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    1, 0, 1, 0
])

#########토큰 수치화######
token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)

print(x)

############패딩##############

padded_x = pad_sequences(
    x,
    maxlen=5,
    padding='pre',#뒤에서 부터는 post
    truncating='post' #디폴트 앞이 짤렸다.
)

print(padded_x)
print(padded_x.shape)# (15, 5)
# [[ 0  0  0  2  3]   >>>>>>>>>>>>>>>>>[0.6,0.7,0.1........0.9] 
#                       input_dim x output_dim 으로 300개의 벡터로 변형됨
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
   

#2.모델
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN

model = Sequential()
# model.add(Embedding(30,100))#순서 input_dim output_dim!! 다른것들은 아웃풋이먼저임
model.add(Embedding(30,100,5))#input_dim output_dim ,input_length(선택적이라 안넣으면 실행안됨)
          #단어 사전의 갯수,  차원 갯수      
model.add(SimpleRNN(10))
model.add(Dense(1))
model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100      
                                                                 
# =================================================================
# Total params: 3,100
# Trainable params: 3,100
# Non-trainable params: 0
# _________________________________________________________________

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

model.fit(
    padded_x,
    labels,
    epochs=100,
    batch_size=32,
    verbose=0
)