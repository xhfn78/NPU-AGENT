#예측해서 나온 결과를 가지고 그걸로 다시 다른 모델로 예측한다 --스태킹
#Pseudo labeling 기법: 모델을 돌려서 나온 결과로 결측치를 찾음
#스태킹: 모델을 돌려 나온거로 컴럼을 구성해서 새로운 데이터셋을 만듬
#     : 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듬

#보팅은 =다수결 import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier
#배깅 모델들
from sklearn.ensemble import StackingClassifier,StackingRegressor,RandomForestClassifier,RandomForestRegressor

from xgboost import XGBClassifier,XGBRFRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
import numpy as np

#1데이터

x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=333,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2-1.모델
xgb = XGBClassifier()
lg = LGBMClassifier(verbosity=-1)
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

model = StackingClassifier(
    estimators=[('XGB', xgb),('LG',lg),('CAT', cat),('RF', rf)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=-1,
    cv=5,
)



#3-2.훈련
model.fit(x_train,y_train)


#4.평가,예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('스태킹 결과:',round(acc,4))

# 스태킹 결과: 0.9649

