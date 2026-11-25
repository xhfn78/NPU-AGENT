import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,r2_score
from xgboost import XGBRegressor
import joblib
import pickle

#배깅은 한가지 모델로 데이터를 여러가지 경우의 수로 훈련시켜서 점수를 도출해냄


#1데이터

x,y = fetch_california_housing(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=333,
    # stratify=y,
    )


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 2000,
              'learning_rate' : 0.1, 
              'max_depth' : 7,
              'gamma': 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 337,
            #   verbose' : 0,
}

#2.모델
path ='./_save/pickle/'
model = pickle.load(open(path+'m20_pickle1_save.pkl','rb'))


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
'''

#원래 점수
# 최종점수: 0.8171235190875237
# r2 0.8171235190875237


#잡립 로드한 점수
# 최종점수: 0.8171235190875237
# r2 0.8171235190875237