#https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

#1. 데이터
# path = './_data/kaggle_santander/'
path = 'c://study//_data//kaggle_santander//'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0 )
# print(test_csv)
submission_csv = pd.read_csv(path +'sample_submission.csv',index_col=0)

# print(train_csv.shape) #(200000, 201)
# print(test_csv.shape) #(200000, 200)
# print(submission_csv.shape) #(200000, 1)

# print(train_csv.isna().sum())  ##결측치 확인 insa(),sum() or isnull(),sum()
# print(test_csv.isna().sum())



x = train_csv.drop(['target'], axis=1)  #타겟 열 한개뺴기
# # print(x) #[10886 rows x 8 columns]

y = train_csv['target']
# print(x.shape,y.shape)  #(200000, 200) (200000,)
# print(y)
# print(np.unique(y,return_counts=True)) 
 #(array([0, 1]), array([179902,  20098]))

x_train,x_test,y_train,y_test =train_test_split(
    x,y,
    random_state=666,
    train_size=0.8,
    stratify=y,  #x,y데이터를 나눌떄 stratify=y이걸안넣으면 x,y서로 데이터 크기가 달랐을떄 비율편차가 생길수있음
)

# #2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=200, activation= 'relu'))
model.add(Dense(80, activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(30, activation= 'relu'))
model.add(Dense(1,activation= 'sigmoid'))




#3.컴파일,훈련
model.compile(loss ='binary_crossentropy',
                optimizer= 'adam',
                metrics=['acc'],        
            )  #이진분류에서는 loss = 'binary_crossetropy' 고정
es = EarlyStopping(
            monitor='val_loss',
            mode= 'auto',
            patience=300,
            restore_best_weights=True,
            )
strat_time = time.time()  #현재 시간을 반환 ,시작시간
hist = model.fit(x_train,y_train,
           epochs=3000 , 
           batch_size=2000,
           validation_split=0.2,
           callbacks =[es], 
           )
end_time = time.time()  #현재 시간을 반환 ,시작시간


#4.평가,예측
loss = model.evaluate(x_test,y_test)
print("loss:", loss)
print('걸린시간 :',round(end_time - strat_time,2),'초')

y_predict = model.predict(x_test)
 #시그모이드 함수를 거쳐 0,1사이 값을 반환후 >>metrics=['acc']로 후처리하면 0 OR 1로 반올림내림해서 퍼센테이지로 변환
y_predict = np.round(y_predict)# y_pred한 값이 0.11121515,0.125148이런식으로 나와서 라운드처리후  [1.] 이런식으로 변환한다음에 acc값 비교 이거 안하면 에러남
acc_score = accuracy_score(y_test,y_predict,normalize=True)
print('acc_score:', acc_score)  #acc_score: 0.9298245614035088

y_submit = model.predict(test_csv)  #test_csv를 pred 에(예측값에 넣고) y_서브밋에 저장
submission_csv['target'] = y_submit # Y_서브밋에 저장된 내용을 서브미션 파일에 "count" 컬럼에 내용 추가 

submission_csv.to_csv(path + 'submit/' + 'submit_0904_3.csv')

# import matplotlib.pyplot as plt
# plt.rc('font', family='Malgun Gothic')  #맑은 고딕 폰트 적용 한글꺠짐 방지
# plt.rcParams['axes.unicode_minus'] = False #마이너스 숫자나올떄 깨짐방지
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'] ,c='red', label='loss') #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'] ,c='blue', label='val_loss')
# plt.legend(loc='upper right') #우측상단에 라벨표시

# plt.title('santander Loss') #제목
# plt.xlabel('epoch') 
# plt.ylabel('loss')
# plt.grid()  #격자표시 추가
# plt.show()

