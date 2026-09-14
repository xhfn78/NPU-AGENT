# [실습] 함수형 모델(Functional API) 맛보기
#
# 지금까지 쓴 Sequential은 "위에서 아래로 한 줄"로만 쌓을 수 있다.
# 함수형 모델은 층을 변수에 담고 괄호로 연결해서 만든다.
#   dense1 = Dense(10)(input1)   ←  input1을 dense1 층에 통과시킨다는 뜻
#
# 왜 굳이 이렇게 쓰나:
#   입력이 두 개거나, 중간에서 갈라졌다 다시 합쳐지는 모델은 Sequential로는 못 만든다.
#   함수형은 연결을 직접 적기 때문에 그런 구조를 만들 수 있다.
#
# 같은 조건이면 순차형과 함수형의 결과는 똑같다. 적는 방식만 다르다.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input


#2-1. 순차적 모델
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

############################################

#2-2. 함수형 모델
input1 = Input(shape=(3,))              # input_shape를 적던 자리를 input1이라는 변수로 지정
dense1 = Dense(10, name='ys1')(input1)  # input1을 이 층에 통과시킨다
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(9, name='ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
output1 = Dense(1)(drop2)

# 마지막에 시작(inputs)과 끝(outputs)을 알려주면 그 사이를 모델로 묶어준다.
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

# name='ys1' 처럼 이름을 붙이면 summary에서 그 이름으로 보인다.
# 층이 많아졌을 때 어떤 층인지 구분하기 편하다.
