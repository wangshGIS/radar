from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

import preprocessing
import network

out_dir = 'output'

def evaluate_radar(model,now_radar_mat, next_radar_mat):
	# evaluate
	print("######model evaluate######")
	#loss, accuracy = model.evaluate(steps=10)#now_radar_mat[::,31:,::,::,::],next_radar_mat[::,31:,::,::,::])#(x_test, y_test) elesun
	result = model.evaluate(now_radar_mat[::,31:,::,::,::],next_radar_mat[::,31:,::,::,::],batch_size=4,verbose=0)
	#print('Test loss:', loss)
	#print 'Test accuracy:', accuracy
	print 'model.metrics_names = ',model.metrics_names
	print 'result = ',result
	print  'val mean_squared_error loss = ',result[0]
	print  'val acc = ',result[2]

def predict_radar(model,now_radar_mat,which):
	# Testing the network on one movie
	# feed it with the first 7 positions and then
	# predict the new positions
	print("######model predict######")

	track = now_radar_mat[which][:31, ::, ::, ::]
	print 'exist track.shape',track.shape  #(31,500,500,1)
	for j in range(30):#30 predict future frames elesun
	    #track_fact = now_radar_mat[which][j:(31+j),::,::,::]#j*batchsize:(31+batch_size) old
	    #track_fact = now_radar_mat[which][31+j:31+j+1,::,::,::]#output1+,seq,acc,notclear,notblack
            #track_fact = now_radar_mat[which][31+j-2:31+j,::,::,::]#output ?,seq,acc,notclear,black
            #track_fact = now_radar_mat[which][31+j-5:31+j+1,::,::,::]#output2
            #output3 37 39error,black ok,but real45 predict46 RAD_32529
            #output4 RAD_32528 black ok,real 45 predict46
            #track_fact = now_radar_mat[which][31+j-6:31+j,::,::,::]#output5 real41 to predict43
            track_fact = now_radar_mat[which][31+j-4:31+j+2,::,::,::]#output6 seq,acc,clear,black 
            #new_pos = model.predict(track[np.newaxis, ::, ::, ::, ::])
	    new_pos = model.predict(track_fact[np.newaxis, ::, ::, ::, ::])
	    print 'model.predict.out_shape',new_pos.shape #(1,31,100,100,1)
	    new = new_pos[::, -1, ::, ::, ::]
	    print 'draw_last_frame.shape',new.shape #(1,100,100,1)
	    track = np.concatenate((track, new), axis=0)  #(32+j-1,100,100,1)+(1,100,100,1)=(32+j,100,100,1)
	    print 'add_',(31+j),'_frames.shape',track.shape
	    #plt.legend()
	    #plt.show()

	# And then compare the predictions
	# to the ground truth
	track_real = now_radar_mat[which][::, ::, ::, ::]
	for i in range(61):#61 frames all elesun
	    fig = plt.figure(figsize=(20, 10))
	    #plt left
	    ax = fig.add_subplot(121)
	    if i == 0:#1 frame
		ax.text(1, 15, 'None', fontsize=20, color='g')
		toplot = np.ones((100,100), dtype=np.float32)#all white png(row,col)
	    elif (0<i<=30):#1 to 31 real frame
		ax.text(1, 15, 'Real_%02d_frame'%(i-1), fontsize=20, color='g')		
		toplot = track_real[i-1, ::, ::, 0]
	    else: #31 to 60 frames
		ax.text(1, 15, 'Predict_%02d_frame'%i, fontsize=20, color='g')
		toplot = track[i, ::, ::, 0]
	    plt.imshow(toplot,cmap='Greys_r')
	    #plt right
	    ax = fig.add_subplot(122)
	    plt.text(1, 15, 'Real_%02d_frame'%i, fontsize=20, color='g')
	    toplot = track_real[i, ::, ::, 0]
	    plt.imshow(toplot,cmap='Greys_r')

	    #save png
	    if not os.path.exists(out_dir):
		   os.mkdir(out_dir)
	    plt.savefig(out_dir+'/'+'%02d_contrast.png' %i)
