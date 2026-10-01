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
    padding='pre',
    truncating='post'
)

print(padded_x)
print(padded_x.shape)       # (15, 5)

##############원핫 인코딩 3가지 만들기#######
##################원 핫1, to_categorycal####################
# from tensorflow.keras.utils import to_categorical
# x_onehot = to_categorical(padded_x).reshape(15,5,31)
# # print(x_onehot)
# # print(x_onehot.shape) #(15, 5, 31)

##################원 핫2, pandas ####################
# x_onehot = pd.get_dummies(padded_x.reshape(-1),dtype=int)
# print(x_onehot)
# print(x_onehot.shape) #(75, 31)
# x_onehot = x_onehot.values.reshape(15,5,31)

# # ##################원 핫3, sklearn ####################
from sklearn.preprocessing import OneHotEncoder
#x_temp = padded_x.reshape(75,1)
x_temp = padded_x.reshape(-1,1)  #(75,1)
# print(x_temp)
# print(x_temp.shape) #(75, 1)
# exit()
# ohe = OneHotEncoder()  #sparse형태로 나온다(혼동행렬)
ohe = OneHotEncoder(sparse_output=False,)
x_onehot = ohe.fit_transform(x_temp)
# print(x_onehot)
# print(x_onehot) #<Compressed Sparse Row sparse matrix of dtype 'float64'
#                with 75 stored elements and shape (75, 31)>

#reshape 조건 1.내용,2순서 (75,)>(75,1)  [1,2,3](3,)> [[1],[2],[3]] (3,1)
x_onehot = x_onehot.reshape(15,5,31)
print(x_onehot)
print(x_onehot.shape) #(15, 5, 31)

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_onehot,
    labels,
    train_size=0.6,
    random_state=333
)

#2모델구성
model = Sequential()

#입력 (5,31)
model.add(LSTM(10, input_shape=(5, 31)))
model.add(Dense(20))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    verbose=0
)

#4.평가,예측
loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

y_test_prob = model.predict(x_test, verbose=0)

# 확률 → 0 또는 1
y_test_predict = (y_test_prob >= 0.5).astype(int).flatten()

acc_score = accuracy_score( y_test,y_test_predict)

print("loss :", loss)
print("keras accuracy :", acc)
print("y_test :", y_test)
print("y_test_predict :", y_test_predict)
print("sklearn accuracy_score :", acc_score)

############새 문장 예측############
x_predict = ['개똥이 잘생겼다']

x_predict_seq = token.texts_to_sequences(x_predict)

x_predict_num = pad_sequences(
    x_predict_seq,
    maxlen=5,
    padding='pre',
    truncating='post'
)

print("새 문장 번호 :", x_predict_num)

##################원 핫1, to_categorycal####################
# x_predict_onehot = to_categorical(x_predict_num,num_classes=31).reshape(1,5,31)
# print(x_predict_onehot)
# print(x_predict_onehot.shape) #(1, 5, 31)

##################원 핫2, pandas ####################
# x_predict_onehot = pd.get_dummies(x_predict_num.reshape(-1),dtype=int)
# print(x_predict_onehot)
# x_predict_onehot = x_predict_onehot.reindex(columns=range(31),fill_value=0)
# x_predict_onehot = x_predict_onehot.values.reshape(1,5,31)

# # ##################원 핫3, sklearn ####################
#x_predict_temp = x_predict_num.reshape(5,1)
x_predict_temp = x_predict_num.reshape(-1,1)  #(5,1)
# print(x_predict_temp)
# print(x_predict_temp.shape) #(5, 1)
x_predict_onehot = ohe.transform(x_predict_temp)
# print(x_predict_onehot)
x_predict_onehot = x_predict_onehot.reshape(1,5,31)
# print(x_predict_onehot.shape) #(1, 5, 31)

prediction = model.predict(x_predict_onehot)

print("긍정 확률 :", prediction)

if prediction >= 0.5:
    print("예측 결과 : 긍정")
else:
    print("예측 결과 : 부정")


 