import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1데이터
df = pd.DataFrame({
    'x1': [1,2,3,4,5],
    'x2': [5,4,3,2,1],
    'x3': [-1000,50,7,100,-1100],
    'y': [2,3,4,5,6],
})

print(df)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(),square=True,annot=True,cbar=True)
plt.show()