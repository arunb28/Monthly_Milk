# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:48:21 2018

@author: Tathagat Dasgupta
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("monthly-milk-production.csv",index_col="Month")
print(data.describe())
print(data.head())

data.index=pd.to_datetime(data.index)  #for converting dates to months
data.plot() #plotting pandas data frame directly without arguments
plt.show()

X_train, X_test = data.head(156) , data.tail(12)

scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

def next_batch(train_data,batch_size,steps):
    batch_start=np.random.randint(0,len(train_data)-(steps+1))
    
    y_batch=np.array(train_data[batch_start:batch_start+steps+1]).reshape(1,steps+1)
    
    return y_batch[:,:-1].reshape(-1,steps,1), y_batch[:,1:].reshape(-1,steps,1) 

tf.reset_default_graph()

num_inputs=1
num_outputs=1
num_time_steps=12
num_neurons=100
learning_rate=0.001
iterations=6000
batch_size=1

x=tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y=tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])

cell=tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)

outputs,states=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

loss=tf.reduce_mean(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for iters in range(iterations):
        x_batch,y_batch=next_batch(X_train,batch_size,num_time_steps)
        
        sess.run(train,feed_dict={x:x_batch,y:y_batch})
        
        if iters %100==0:
        
            mse=loss.eval(feed_dict={x:x_batch,y:y_batch})
            print(iters, "\tMSE",mse)
        
    saver.save(sess,"/RNN_models/RNN_milk_two")
            