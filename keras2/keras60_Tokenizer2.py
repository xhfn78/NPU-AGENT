from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


text1 = ' 나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄청 마구 마구 마구 마구 먹었다.'
text2 = ' 개똥이는 기관사를 좋아한다. 말똥이는 잘생겼다. 길동이는 마구마구 더 잘생겼다.'
token = Tokenizer() # token=객체(인스턴스)  = Tokenizer 클래스의 인스턴스 생성
#단어 빈도를 세고 자주 나온 단어부터 번호를 붙임
token.fit_on_texts([text1,text2])

#단어와 번호 사전 확인
print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '잘생겼다': 4, '나는': 5, '지금': 6, '맛있는': 7, '김밥을': 8, 
#  '엄청': 9, '먹었다': 10, '개똥이는': 11, '기관사를': 12, '좋아한다': 13, '말똥이는': 14, '길동이는': 15, '마구마구': 16, '더': 17}#########토큰 수치화######
#문장의 단어를 번호 리스트로 변환
x = token.texts_to_sequences([text1,text2])  #두개의 리스트가 나옴
#길이가 다른 두 문장 리스트를 1차원으로 합침 (23,)
x = np.concatenate(x) #두개의 리스트를 합쳐서 하나의 리스트로 만듦

print(x) #두 문장의 번호를 합친 1차원 배열


# x = np.array(x)# 리스트로 된 x를 numpy array로 변환
print(x.shape) #(23,), 두 문장을 합친 단어 수
###########토큰 원핫인코딩으로 수치화############
##############원핫 인코딩 3가지 만들기#######

##################원 핫1, to_categorycal####################
#0번 열도 생성, 두 문장 데이터는 (23,18)로 reshape 값을 맞춤
# from tensorflow.keras.utils import to_categorical
# x = to_categorical(x).reshape(14,10) #(14, 10)
# print(x)
# print(x.shape)
##################원 핫2, pandas ####################
#번호를 1차원으로 펴고 실제 나온 번호마다 열 생성
# x = pd.get_dummies(x.reshape(-1),dtype=int)
# print(x.shape) #(23, 17)
# # ##################원 핫3, sklearn ####################
from sklearn.preprocessing import OneHotEncoder
x = x.reshape(-1,1)  #(23,1), sklearn 입력은 2차원
ohe = OneHotEncoder()  #기본 출력은 sparse형태(희소행렬)
ohe = OneHotEncoder(sparse_output=False) #원핫 결과를 일반 배열로 반환
x = ohe.fit_transform(x) #단어 번호를 0,1 벡터로 변환 (23,17)
print(x)
print(x.shape) #(23, 17)  
#reshape 조건 1.내용,2순서 (23,)>(23,1)  [1,2,3](3,)> [[1],[2],[3]] (3,1)
# reshape의 -1은 "나머지는 알아서 맞춰라"라는 뜻이다.
#
# 원-핫 방법 3가지 비교
#   to_categorical (텐서플로) : 0번 클래스가 없어도 0번 자리를 만들어버리는 함정이 있다
#   get_dummies (판다스)      : 존재하는 클래스에 대해서만 열을 만든다. dtype=int와 .values를 붙이는 게 안전
#   OneHotEncoder (사이킷런)  : 2차원 입력을 받으므로 reshape(-1,1)이 필요하다
#
#원핫 방식은 사용할 부분만 주석을 해제해서 선택
