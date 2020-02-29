""" 
radar_train_test
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import psutil

import preprocessing
import network
import model_evaluate_predict

mode = 'test' # train or test
epochs = 10#100
batch_size = 6#6
n_samples = 1#train for 9;test for 1
n_frames=61
row = 501
col = 501
which = 0#<n_samples,which samples you select to predict elesun


model_dir = 'model'
model_file = os.path.join(model_dir, 'model_train_radar.h5')

time1 = time.time()
# preprocessing data
print("######preprocessing data######")
print 'pid memory used = ',round((psutil.Process(os.getpid()).memory_info().rss)/1024.0/1024.0/1024.0*10,2),'Gbytes'
now_radar_mat, next_radar_mat = preprocessing.generate_radar_data(n_samples,n_frames,row,col)
print 'now_radar_mat.shape',now_radar_mat.shape
print 'now_radar_mat.dtype',now_radar_mat.dtype
#print(now_radar_mat)
print 'next_radar_mat.shape',next_radar_mat.shape
print 'next_radar_mat.dtype',next_radar_mat.dtype
#print(next_radar_mat)
print 'pid memory used = ',round((psutil.Process(os.getpid()).memory_info().rss)/1024.0/1024.0/1024.0*10,2),'Gbytes'
time2 = time.time()
print ('time use:' + str(time2 - time1) + 's')

print('################'+mode+'################')
time3 = time.time()
if(mode=='train'):
	model = network.network(row,col)
	#history = model.fit(now_radar_mat[::,:61,::,::,::], next_radar_mat[::,:61,::,::,::], batch_size=batch_size,epochs=epochs, validation_split=0.05)
	result = np.empty([0,3])
	for epoch in range(epochs) :
		print '****************epoch %d/epochs %d****************'%(epoch,epochs)	
		for step_samples in range(n_samples) :
			#print '********step_sample %d/n_samples %d********'%(step_samples,n_samples)
			steps_frame = n_frames/batch_size
			#print 'steps_frame=',steps_frame
			for next_batch in range(steps_frame):
				#print '****next_batch %d/steps_frame %d****'%(next_batch,steps_frame)		
				history = model.train_on_batch(\
	now_radar_mat[step_samples,(next_batch*batch_size):(next_batch+1)*batch_size,::,::,::]\
	.reshape(1,batch_size,row,col,1), \
	next_radar_mat[step_samples,(next_batch*batch_size):(next_batch+1)*batch_size,::,::,::]\
	.reshape(1,batch_size,row,col,1),\
	class_weight=None, sample_weight=None)
				#print 'history = ',history
				print 'training epoch: %04d,step_sample: %02d,batch: %02d,loss: %.4f,acc: %.4f'%(epoch,step_samples,next_batch,history[0],history[2])
		print 'pid memory used = ',\
	round((psutil.Process(os.getpid()).memory_info().rss)\
	/1024.0/1024.0/1024.0*10,2),'Gbytes'
		result = np.row_stack((result,history))
		print 'result = ',result
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)
	model.save(model_file)#elesun
	print('Model Saved.')#elesun
	# plot history
	plt.plot(np.arange(epochs),result[:,0], label='train_loss')
	plt.plot(np.arange(epochs),result[:,2], label='train_acc')
        plt.xlabel('epochs')
        plt.ylabel('loss or acc')
	plt.legend()
	plt.savefig('loss.png')
	#plt.show()
elif(mode=='test'):
	# load the trained model
	print('#load the trained model:')
	if not os.path.isfile(model_file):
	    print(model_file+" not exist!")
	    exit(0)
	model = load_model(model_file)
        model_evaluate_predict.evaluate_radar(model,now_radar_mat, next_radar_mat)
        model_evaluate_predict.predict_radar(model,now_radar_mat,which)
else :
	print('#there is no your mode! tips:mode=train or test')
	exit(0)
print 'pid memory used = ',round((psutil.Process(os.getpid()).memory_info().rss)/1024.0/1024.0/1024.0*10,2),'Gbytes'
time3 = time.time()
print ('time use:' + str(time3 - time2) + 's')

