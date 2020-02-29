# -*- coding: utf-8 -*-
"""
generate_radar_data
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img_dir = "./data/"

#generate_radar_data
def generate_radar_data(n_samples,n_frames,row,col):

	now_radar_mat = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float32)
	next_radar_mat = np.zeros((n_samples, n_frames, row, col, 1),dtype=np.float32)
	print (now_radar_mat.shape)
	print (now_radar_mat.dtype)
	#print (now_radar_mat)


	file_list = os.listdir(img_dir)
	file_list.sort(key=lambda x:int(x[4:])) #RAD_32530 RAD_32531
	#print(file_list)
	filecnt = 0
	for filename,filecnt in zip(file_list,range(n_samples)):	
		path = ''
		path = img_dir+filename
		img_list = os.listdir(path)
		img_list.sort(key=lambda x:int(x[20:-4]))#RAD_206582404232531_000.png  RAD_206582404232531_001.png
		#print(img_list)
		imgcnt = 0
		for imgname in img_list:
		
			path = ''
			path = img_dir+filename+"/"+imgname
			img = mpimg.imread(path) # 读取和代码处于同一目录下的png
			# 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
			#print (img.shape) #(50, 50)
			#print (img.dtype)
			#print (img)
			now_radar_mat[filecnt][imgcnt][::,::,0] = img
			if imgcnt < n_frames-1:
				next_radar_mat[filecnt][imgcnt+1][::,::,0] = img
			imgcnt = imgcnt + 1
		filecnt = filecnt + 1
	return now_radar_mat, next_radar_mat



