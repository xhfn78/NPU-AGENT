import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))  #시그모이드 수식

sigmoid = lambda x: 1/ (1+np.exp(-x))

x = np.arange(-5,5,0.1)  # -5부터 5까지 0.1로 찍은데이터 나옴
print(x)
print(len(x)) #100개 나옴

y = sigmoid(x)


plt.plot(x,y)
plt.grid()
plt.show()  #시그모이드는 0에서 1사이값 한정해주는 activation 또는 한정 함수