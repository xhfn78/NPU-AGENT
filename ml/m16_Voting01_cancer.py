#보팅은 =다수결 import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier
#배깅 모델들
from sklearn.ensemble import VotingClassifier,VotingRegressor,RandomForestClassifier

from xgboost import XGBClassifier,XGBRFRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier,LGBMRegressor



#1데이터

x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=333,
    stratify=y,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
xgb = XGBClassifier()
lg = LGBMClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()


model = VotingClassifier(
    estimators=[('XGB',xgb),('RF',rf),('LG',lg),('CAT',cat)],
    # voting='hard',#hard가 디폴트임
    voting='soft',
    
)



# model = DecisionTreeClassifier()
# model = BaggingClassifier(
#     DecisionTreeClassifier(),
#     n_estimators=1000,#epochs와 비슷 현재 여기서는  DecisionTreeRegressor 모델 10번 돌림
#     random_state=333,
#     n_jobs=-1,
#     max_samples=0.8, #train 데이터 비율
#     bootstrap=True #디폴트 /이터 중복허용
#     bootstrap=False #중복 허용 x
# )

# model = RandomForestClassifier()

#3.훈련

model.fit(x_train,y_train)

#4.평가예측

result = model.score(x_test,y_test)
print('최종점수:',result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('acc', round(acc,2))

'''
#디시젼 트리 점수
최종점수: 0.9122807017543859
acc 0.91


배깅(bootstrap=True) <<<통상적으로는 중복허용한게 더좋을지도?
최종점수: 0.9824561403508771
acc 0.98


배깅(bootstrap=Flase)
최종점수: 0.9912280701754386
acc 0.99

랜덤 포레스트
최종점수: 0.9824561403508771
acc 0.98





hard보팅
최종점수: 0.9824561403508771
acc 0.98

soft보팅
최종점수: 0.9912280701754386
acc 0.99

'''