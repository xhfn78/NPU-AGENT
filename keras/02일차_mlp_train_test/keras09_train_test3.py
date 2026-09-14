import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 앞부분/뒷부분으로 슬라이싱해서 나눈 방법(train_test2)은 잘못되었다.
# 데이터의 범위는 전체 범위인데, 앞부분만으로 훈련해서 뒷부분을 평가하는 셈이기 때문이다.
# (다른 범위로 훈련하고 또 다른 범위로 테스트했으니 w 기울기에 오차가 커지는 게 당연하고,
#  그렇게 나온 평가도 신뢰할 수 없다.)
# 훈련도 전체 범위에서, 테스트도 전체 범위에서, 즉 같은 범위 안에서 랜덤으로 나눠야 한다.
# 그래야 평가를 믿을 수 있다.
#
# 그 일을 해주는 것이 사이킷런의 train_test_split 이다.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7,   # 같이 써도 상관없고, 통상적으로 하나만 써도 된다
    # test_size=0.3,
    #   - test_size=0.3 만 적는 것과 train_size=0.7 만 적는 것은 같다.
    #   - test_size=0.3, train_size=0.8 처럼 합이 1을 넘으면 에러가 난다.
    #   - test_size=0.2, train_size=0.7 처럼 합이 1보다 작은 건 실행된다.
    #   - train_size 디폴트 0.75 / test_size 디폴트 0.25

    # shuffle=True,     # 디폴트로 True(섞는다)가 들어간다

    # 고정하지 않으면 실행시킬 때마다 나뉘는 값이 달라진다 >>>
    random_state=444,   # 난수표의 값을 지정해서 랜덤 분할을 고정시킨다 (랜덤시드라고도 부른다)
    # random_state가 고정이면 훈련/테스트 데이터가 매번 동일하게 뽑힌다.
    # 대회 데이터에서는 random_state만 잘 맞춰도 성적이 크게 달라지는 경우가 있다.
)

print('x_train :', x_train)
print('x_test  :', x_test)
print('y_train :', y_train)
print('y_test  :', y_test)

# x_train : [ 2  9  4  3  1 10  5]
# x_test  : [7 8 6]
# y_train : [ 2  9  4  3  1 10  5]
# y_test  : [7 8 6]

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
results = model.predict(np.array([11]))
print('11의결과값: ', results)

#랜덤값=333
# loss: 0.09532500058412552
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
# 11의결과값:  [[10.784775]]

#랜덤값=444
# loss: 0.14206752181053162
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step
# 11의결과값:  [[10.672892]]
