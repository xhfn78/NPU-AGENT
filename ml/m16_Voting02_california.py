#보팅은 =다수결 import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier
#배깅 모델들
from sklearn.ensemble import VotingClassifier,VotingRegressor,RandomForestClassifier,RandomForestRegressor

from xgboost import XGBClassifier,XGBRFRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from lightgbm import LGBMClassifier,LGBMRegressor

#회귀문제는 하드 보팅이없음 왜나하면 0,1,2냐 가 중요하지않기 때문에

#1데이터

x,y = fetch_california_housing(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=42,
    # stratify=y,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
xgb = XGBRFRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()


model = VotingRegressor(
    estimators=[('XGB',xgb),('RF',rf),('LG',lg),('CAT',cat)],
)


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
최종점수: 0.5987728709773144
r2 0.5987728709773144


배깅(bootstrap=True) <<<통상적으로는 중복허용한게 더좋을지도?
최종점수: 0.7709319970417915
r2 0.7709319970417915


배깅(bootstrap=Flase)<<<<이게 젤좋음
최종점수: 0.771260941840586
r2 0.771260941840586



최종점수: 0.8081268277409894
r2 0.8081268277409894



랜덤포레스트 
최종점수: 0.812647430358174
r2 0.812647430358174


리그레서 보팅
최종점수: 0.8196905744115315
r2 0.8196905744115315

최종점수: 0.819776395246794
r2 0.819776395246794

최종점수: 0.8204378507651134
r2 0.8204378507651134


'''