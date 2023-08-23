#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 00:24:44 2023

@author: ashleymuoki
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB

#read the data 
#but first change working directory in the console 
df = pd.read_csv("Golf play days.csv")

#display the data
print(df.head())

#show information on the data

print(df.info())

#convert data into categorical data 
df = df.apply(lambda x:x.astype('category'))

#create a dataframe in which you have everything as a category 

df1 = df.apply(lambda x:x.cat.codes)

#split the dataframe 

train = df1[:10]
test  = df1[-4:]

x_train = train
y_train = train.pop('Play')

x_test = test
y_test =  test.pop('Play')


#initialze your model
model = MultinomialNB()

#fit your data 
model_obj = model.fit(x_train, y_train)

#carry out prediction ie which days play is happening

y_pred = model.predict(x_test)

#check accuracy
print("Training accuracy",model.score(x_train,y_train))
print("Testing accuracy", model.score(x_test, y_test))



