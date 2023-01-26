


import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")




X=data.drop('target',axis=1)
y=data.pop('target')


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler() 
X_train_ss=ss.fit_transform(X_train)
X_test_ss=ss.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr_model=lr.fit(X_train_ss,y_train)

pickle.dump(lr_model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))







