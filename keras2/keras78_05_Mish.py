import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)


def mish(x):
    return x * np.tanh(np.log(1+np.exp(x)))  #log 변환이 들어가서 연산량이 >silu에 비해서 많음 

mish(x) = lambda x : x * np.tanh(np.log(1+np.exp(x)))  

y =mish(x)

plt.plot(x,y)
plt.grid()
plt.show()


#relu에서 0이하의 값을 무시해보니까 뭔가 찜찜해서 음수도 살려봄