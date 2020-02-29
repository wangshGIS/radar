
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
def network(row,col):
		   
	model = Sequential()
	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
					   input_shape=(None, row, col, 1),
					   padding='same', return_sequences=True))
	model.add(BatchNormalization())

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
					   padding='same', return_sequences=True))
	model.add(BatchNormalization())

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
					   padding='same', return_sequences=True))
	model.add(BatchNormalization())

	model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
					   padding='same', return_sequences=True))
	model.add(BatchNormalization())

	model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
				   activation='sigmoid',
				   padding='same', data_format='channels_last'))
	#model.summary()
	#model.compile(loss='mean_squared_error', optimizer='RMSprop')
	model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['mse', 'acc'])

	return model
