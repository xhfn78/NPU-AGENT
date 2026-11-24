import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
alpha= 1

# def elu(x,alpha):
#     return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))


elu = lambda x,alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))
y =elu(x,alpha) #알파값 1이하로 안떨어짐

plt.plot(x,y)
plt.grid()
plt.show()


#relu에서 0이하의 값을 무시해보니까 뭔가 찜찜해서 음수도 살려봄