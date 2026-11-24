#예측해서 나온 결과를 가지고 그걸로 다시 다른 모델로 예측한다 --스태킹
#Pseudo labeling 기법: 모델을 돌려서 나온 결과로 결측치를 찾음
#스태킹: 모델을 돌려 나온거로 컴럼을 구성해서 새로운 데이터셋을 만듬
#     : 한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼을 만듬

#보팅은 =다수결 import numpy as np
from sklearn.datasets import load_diabetes
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

x,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=333,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2-1.모델
xgb = XGBRFRegressor()
lg = LGBMRegressor(verbosity=-1)
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

models = [xgb,lg,rf,cat]
train_list= []
test_list= []


for model in models:
    #3-1.훈련
    model.fit(x_train,y_train)
    y_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_pred)
    test_list.append(y_test_pred)

    r2 = r2_score(y_test,y_test_pred)
    class_name = model.__class__.__name__
    print('{0}acc:{1:.4f}'.format(class_name,r2))


# XGBRFRegressoracc:0.3528
# LGBMRegressoracc:0.3441
# RandomForestRegressoracc:0.3599
# CatBoostRegressoracc:0.3786


x_train_new = np.array(train_list).T
# print(x_train_new.shape)# 
x_test_new = np.array(test_list).T
# print(x_test_new.shape) 

#3-2모델
model2 = CatBoostRegressor(verbose=0)


#3-2.훈련
model2.fit(x_train_new,y_train)


#4.평가,예측
y_pred_new = model2.predict(x_test_new)
r2_new = r2_score(y_test,y_pred_new)
print('스태킹 결과:',round(r2_new,2))

# 스태킹 결과: 0.36

