# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:39:10 2018

@author: Tathagat Dasgupta
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("monthly-milk-production.csv",index_col="Month")
data.index=pd.to_datetime(data.index)  #for converting dates to months
X_train, X_test = data.head(156) , data.tail(12)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

tf.reset_default_graph()

num_inputs=1
num_outputs=1
num_time_steps=12
num_neurons=100


x=tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y=tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])

cell=tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)
outputs,states=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

print(X_test)

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"/RNN_models/RNN_milk_two")
    
    train_seed=list(X_train[-12:])
    
    for iters in range(12):
        X_batch= np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        
        y_pred=sess.run(outputs,feed_dict={x:X_batch})
        
        train_seed.append(y_pred[0,-1,0])
        
results=scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

X_test=pd.DataFrame(X_test)
X_test['Generated']= results

print(X_test)

X_test.plot()
plt.show()
