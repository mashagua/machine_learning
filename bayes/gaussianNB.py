# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:52:31 2018

@author: masha
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
data=pd.read_csv("I:/python_project/machine_learning/bayes/career_data.csv")
data["985_cleaned"]=np.where(data["985"]=="Yes",1,0)
data["edu_cleaned"]=np.where(data["education"]=="bachlor",1,
    np.where(data["education"]=="master",2,
             np.where(data["education"]=="phd",3,4)))

data["skill_cleaned"]=np.where(data["skill"]=="c++",1,np.where(data["skill"]=="java",2,3))
data["enrolled_cleaned"]=np.where(data["enrolled"]=="Yes",1,0)

X_train, X_test = train_test_split(data, test_size=0.1,
                                   random_state=int(time.time()))
gnb=GaussianNB()
used_features=[
        "985_cleaned",
        "edu_cleaned",
        "skill_cleaned"]
gnb.fit(X_train[used_features].values,
        X_train["enrolled_cleaned"])
y_pred=gnb.predict(X_test[used_features])
print((1-(X_test["enrolled_cleaned"]!=y_pred).sum()/X_test.shape[0]))