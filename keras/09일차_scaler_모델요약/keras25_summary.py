# [실습] model.summary()로 모델 구조와 파라미터 개수 확인하기
#
# Param # 계산법: (입력 노드 수 × 출력 노드 수) + 출력 노드 수(bias)
#   dense   : 입력 1 → 출력 3  →  (1×3) + 3 = 6
#   dense_1 : 입력 3 → 출력 4  →  (3×4) + 4 = 16
#   dense_2 : 입력 4 → 출력 3  →  (4×3) + 3 = 15
#   dense_3 : 입력 3 → 출력 1  →  (3×1) + 1 = 4
#   합계 41개. 이 41개가 훈련으로 조정되는 w와 b의 총 개수다.
#
# Output Shape의 None은 행(샘플 개수)을 뜻한다.
# 몇 개가 들어올지는 미리 정해지지 않으므로 None으로 둔다. (행 무시 열 우선)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 



#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim= 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
'''
 Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 3)                   │               6 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 4)                   │              16 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 3)                   │              15 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 1)                   │               4 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 41 (164.00 B)
 Trainable params: 41 (164.00 B)
 Non-trainable params: 0 (0.00 B)
'''