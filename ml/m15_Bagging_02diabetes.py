import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
#배깅 모델들
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor


#1데이터

x,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=333,
    # stratify=y,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
# model = DecisionTreeRegressor()
model = BaggingRegressor(
    DecisionTreeRegressor(),
    n_estimators=1000,#epochs와 비슷 현재 여기서는  DecisionTreeRegressor 모델 10번 돌림
    random_state=333,
    n_jobs=-1,
    max_samples=0.8, #train 데이터 비율
    bootstrap=True #디폴트 /이터 중복허용
    # bootstrap=False #중복 허용 x
)

# model = RandomForestRegressor()

#3.훈련

model.fit(x_train,y_train)

#4.평가예측

result = model.score(x_test,y_test)
print('최종점수:',result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2', r2)

'''
#디시젼 트리 점수
최종점수: -0.04779426074816895
r2 -0.04779426074816895


배깅(bootstrap=True) <<<통상적으로는 중복허용한게 더좋을지도?
최종점수: 0.39764017682901676
r2 0.39764017682901676


배깅(bootstrap=Flase)<<<<이게 젤좋음
최종점수: 0.34213672375795745
r2 0.34213672375795745

랜덤 포레스트
최종점수: 0.39310665988703597
r2 0.3931066598870

'''