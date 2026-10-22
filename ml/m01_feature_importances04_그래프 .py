from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

#1 데이터

x,y = load_iris(return_X_y=True)#이렇게 하면 바로 xy 분리됨 사이킷런 데이터 한정

print(x.shape,y.shape)

random_state = 115

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,
    train_size=0.8,
    stratify=y,
)

#2.모델구성

model= XGBClassifier(random_state=random_state)

model.fit(x_train,y_train)
print('================',model.__class__.__name__,'========')
#4.평가,예측
score = model.score(x_test,y_test)
print('acc:', score)
print(model.feature_importances_)
#F 스코어: 빈도수
gain_score = model.get_booster().get_score(importance_type='gain')
print('게인 스코어',gain_score)#게인 스코어가 높을 수록 좋음
#gain 스코어 :손실함수의 개선점수 ,높으면 개선 많이됨

import matplotlib.pyplot as plt
from xgboost import plot_importance

#F-score 그래프 
# plot_importance(model,importance_type ='weight')
# plt.title('F-score')
# plt.show()
plot_importance(model,importance_type ='gain')
plt.title('F-score')
plt.show()