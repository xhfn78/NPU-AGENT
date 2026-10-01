import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다','한 번 더 보고 싶어요','글쎄',
    '별로에요','생각보다 지루해요','연기가 어색해요',
    '재미없어요','너무 재미없다','참 재밌네요',
    '개똥이 바보','말똥이 잘생겼다','길동이 또 구라친다',
    ]

#문장별 정답, 긍정 1 / 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0]) 
print(labels.shape) #(15,)

token = Tokenizer()
#단어 빈도를 세고 자주 나온 단어부터 번호를 붙임
token.fit_on_texts(docs)
# print(token.word_index)
{'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, 
 '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
 '재미없어요': 21, '재미없다': 22, '재밌네요': 23, '개똥이': 24, '바보': 25, '말똥이': 26, '잘생겼다': 27, '길동이': 28, '또': 29, '구라친다': 30}

#문장의 단어를 번호 리스트로 변환
x = token.texts_to_sequences(docs)

print(x)
#  [[2, 3], [1, 4], 
#  [1, 5, 6], [7, 8, 9],
#  [10, 11, 12, 13, 14], 
#  [15], [16], [17, 18], 
#  [19, 20], [21], [2, 22],
#  [1, 23], [24, 25], 
#  [26, 27], [28, 29, 30]]
#각 단어마다 쉐이프가 다름 
############패딩##############
from tensorflow.keras.preprocessing.sequence import pad_sequences
#문장 길이를 5로 맞춤, 짧으면 앞에 0을 채우고 길면 뒤를 자름
padded_x = pad_sequences(x,
                         padding= 'pre', #앞을 0으로 채울려면 pre, 뒤는 post
                         maxlen=5, 
                         truncating='post' #디폴트 앞이 짤림
                         )
# print(padded_x)
# [[ 0  0  0  2  3]
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
#  [ 0  0 28 29 30]] 제일 큰 5개에 맞춰서 앞이 0으로 채워짐
print(padded_x.shape) #(15, 5)

#학습 60%, 테스트 40%로 문장과 정답을 함께 분리
x_train,x_test,y_train,y_test = train_test_split(padded_x,labels,train_size=0.6,random_state=333,)

#2모델구성
#DNN은 문장별 번호 5개를 2차원 입력으로 받음 (15,5)
model = Sequential()
model.add(Dense(10, input_shape=(5,)))
model.add(Dense(20))
model.add(Dense(30,activation='relu'))
model.add(Dense(1 ,activation='sigmoid'))

#3컴파일 훈련
#이진분류 손실과 adam 설정, acc로 정확도 확인
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'],)
#학습 데이터를 100번 반복, 배치 크기 32
model.fit(x_train,y_train,epochs=100,batch_size=32,)

#4.평가,예측
loss = model.evaluate(x_test,y_test) #테스트 평가, [loss, acc] 반환
print('==============================')
print("loss:", loss[0])
print('acc:',round(loss[1],4))
print('==============================')
y_pred = model.predict(x_test)  #시그모이드 함수를 거쳐 0,1사이 값을 반환
y_pred = np.round(y_pred) #예측 확률을 0 또는 1로 반올림
print(y_pred)
# print(y_pred)
acc_score = accuracy_score(y_test,y_pred,normalize=True) #테스트 정답과 예측값 비교
print('acc_score:', acc_score)

#새 문장도 학습에 사용한 사전으로 변환
x_predict = ['개똥이 잘생겼다']
x_predict_seq = token.texts_to_sequences(x_predict) #새 문장 -> 단어 번호
#새 문장도 길이 5, 학습과 같은 패딩 적용
x_predict_num = pad_sequences(x_predict_seq,maxlen=5,padding='pre',truncating='post',)
y_predict = model.predict(x_predict_num) #새 문장의 긍정 확률 예측

print('개똥이 잘생겼다의 결과 :', y_predict)
