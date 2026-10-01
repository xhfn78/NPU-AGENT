import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다', '참 재밌네요',
    '개똥이 바보', '말똥이 잘생겼다', '길동이 또 구라친다',
]

#문장별 정답, 긍정 1 / 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

#########토큰 수치화######
token = Tokenizer()
#단어 빈도를 세고 자주 나온 단어부터 번호를 붙임
token.fit_on_texts(docs)

#문장의 단어를 번호 리스트로 변환
x = token.texts_to_sequences(docs)

############패딩##############
#문장 길이를 5로 맞춤, 짧으면 앞에 0을 채우고 길면 뒤를 자름
padded_x = pad_sequences(x,maxlen=5,padding='pre',truncating='post',)

print(padded_x.shape) #(15, 5)

#LSTM은 3차원으로 들어감 (15,5) -> (15,5,1)
padded_x = padded_x.reshape(15,5,1)

print(padded_x.shape) #(15, 5, 1)

#train_test_split
#학습 60%, 테스트 40%로 문장과 정답을 함께 분리
x_train,x_test,y_train,y_test = train_test_split(padded_x,labels,train_size=0.6,random_state=333,)

#2모델구성
#LSTM은 번호 5개를 순서대로 읽음, 단어당 특성 1개 (15,5,1)
model = Sequential()
model.add(LSTM(512, input_shape=(5,1)))
model.add(Dense(20))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3컴파일 훈련
#이진분류 손실과 adam 설정, acc로 정확도 확인
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'],)

#학습 데이터를 100번 반복, 배치 크기 32
model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=0,)

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
x_predict_num = pad_sequences(x_predict_seq,
                              maxlen=5,
                              padding='pre',
                              truncating='post',)

#LSTM 입력에 맞춰서 3차원으로 변경
x_predict_num = x_predict_num.reshape(1,5,1)
y_predict = model.predict(x_predict_num) #새 문장의 긍정 확률 예측

print('개똥이 잘생겼다의 결과 :', y_predict)
