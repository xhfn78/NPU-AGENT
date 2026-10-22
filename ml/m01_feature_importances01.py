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

#모델 전체 실행

model.fit(x_train,y_train)
print('================',model.__class__.__name__,'========')
#4.평가,예측
score = model.score(x_test,y_test)
print('acc:', score)
print(model.feature_importances_)


# ================ DecisionTreeClassifier ========
# acc: 0.9666666666666667
# [0.0075   0.01875  0.588772 0.384978] (150, 4) (150,) 피처에 대한 중요도 나옴
# ================ RandomForestClassifier ========
# acc: 0.9666666666666667
# [0.12180241 0.03897144 0.45046012 0.38876603]
# ================ GradientBoostingClassifier ========
# acc: 0.9666666666666667
# [0.00374113 0.01546393 0.29725693 0.68353801]
# ================ XGBClassifier ========
# acc: 0.9666666666666667
# [0.02116161 0.02738614 0.6533406  0.29811168]
