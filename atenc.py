from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, Bidirectional, Concatenate, dot, Activation, TimeDistributed
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from attention import AttentionLayer
from keras import backend as k
from keras import regularizers
#from custom_recurrents import AttentionDecoder
import numpy as np
import sys
import time


def create_model(latent_dim, bidirectional):
	
	## Create encoder
	encoder_inputs = Input(shape=(None, 14))

	if bidirectional:
		encoder = Bidirectional(LSTM(latent_dim, return_state = True))
		encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
		state_h = Concatenate()([forward_h, backward_h])
		state_c = Concatenate()([forward_c, backward_c])
		print('state_h.size', k.shape(state_h))
		print('state_c.size', k.shape(state_c))
	else:
		encoder = LSTM(latent_dim, return_sequences=True, return_state = True)
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		#encoder_out = encoder(encoder_inputs)
	
	encoder_states = [state_h, state_c]
	encoder_out = [encoder_outputs, state_h, state_c]
	## Keep encoder as separate model
	enc_model = Model(encoder_inputs,encoder_out,name='encoder_model')


	## Create decoder
	decoder_input = Input(shape=(None, 38))
	if bidirectional:
		in_h_state = Input(shape=(2*latent_dim,))
		in_e_state = Input(shape=(2*latent_dim,))
		decoder_lstm = LSTM(latent_dim*2, return_sequences = True, return_state = True)
	else:
		in_h_state = Input(shape=(latent_dim,))
		in_e_state = Input(shape=(latent_dim,))
		decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
	decoder_outputs, dec_h_state, dec_c_state = decoder_lstm(decoder_input, initial_state = [in_h_state,in_e_state])

	attn_enc = Input(shape=(None, latent_dim))
	attention = dot([decoder_outputs, attn_enc], axes=[2, 2])
	attention = Activation('softmax')(attention)

	context = dot([attention, attn_enc], axes=[2,1])
	decoder_combined_context = Concatenate()([context, decoder_outputs])

	# equation (5) of the paper
	output = TimeDistributed(Dense(latent_dim, activation="tanh"))(decoder_combined_context)

	decoder_dense = Dense(38,activation='softmax', activity_regularizer=regularizers.l2(0.05))
	#decoder_outputs = decoder_dense(decoder_outputs)
	output = decoder_dense(output)
	## Keep decoder as separate model
	dec_model = Model([decoder_input,attn_enc,in_h_state,in_e_state],[output,dec_h_state,dec_c_state],name='decoder_model')

	## Combined encoder and decoder model
	## Inputs are encoder input and decoder input
	## Output is the 0th output of dec_model (not the states) when applied on decoder input with encoded states
	model = Model([encoder_inputs,decoder_input],dec_model([decoder_input]+enc_model(encoder_inputs))[0],name='combined_model')
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	
	## Return all 3 models
	enc_model.summary()
	dec_model.summary()
	model.summary()
	#input('pause')
	return enc_model,dec_model,model


def train():
	max_len = 128
	batch_size = 64
	epochs = 300
	latent_dim = 128
	bidirectional = False
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(suppress=True)
	path = '../../../../data/data1/users/dsyrrafos/seq2seq/dataset/'

	#encoder_input_data = np.load(path+'encoder_input_data.npy')[:,:max_len,:]
	#decoder_input_data = np.load(path+'decoder_input_data.npy')[:,:max_len,:]
	#decoder_target_data = np.load(path+'decoder_target_data.npy')[:,:max_len,:]
	
	encoder_input_data = np.load(path+'small_encoder_input_data.npy')
	decoder_input_data = np.load(path+'small_decoder_input_data.npy')
	decoder_target_data = np.load(path+'small_decoder_target_data.npy')

	path = '../../../../data/data1/users/dsyrrafos/seq2seq/logs/'
	encoder,decoder,combined = create_model(latent_dim, bidirectional)
	tensorboard = TensorBoard(log_dir=path+"logs/{}".format(time.time())+'_'+str(epochs))

	plot_model(combined, to_file='combined.png', show_shapes=True)
	plot_model(encoder, to_file='encoder.png', show_shapes=True)
	plot_model(decoder, to_file='decoder.png', show_shapes=True)

	combined.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[tensorboard])

	path = '../../../../data/data1/users/dsyrrafos/seq2seq/models/'
	encoder.save(path+'encoder.h5')
	decoder.save(path+'decoder.h5')
	combined.save(path+'combined.h5')


def predict():	
	max_len = 128

	test_song = 4
	path = '../../../../data/data1/users/dsyrrafos/seq2seq/models/eddie/'
	encoder = load_model(path+'encoder.h5')
	decoder = load_model(path+'decoder.h5')

	encoder.summary()
	decoder.summary()
	path = '../../../../data/data1/users/dsyrrafos/seq2seq/dataset/'
	encoder_input_data = np.load(path+'test_encoder_input_data.npy')[test_song:test_song+1,:max_len,:]
	decoder_input_data = np.load(path+'test_decoder_input_data.npy')[test_song:test_song+1,:max_len,:]
	decoder_target_data = np.load(path+'test_decoder_target_data.npy')[test_song:test_song+1,:max_len,:]

	enc_h,enc_c = encoder.predict(encoder_input_data)
	target_seq = np.zeros((1, 1, 38))
	target_seq[0,0,36] = 1

	o_token,h,c = decoder.predict([target_seq,enc_h,enc_c])
	
	target_seq=np.concatenate([target_seq,o_token],axis=1)

	for i in range(max_len):
		print(i)
		o_token,h,c = decoder.predict([o_token,h,c])
		target_seq=np.concatenate([target_seq,o_token],axis=1)

	print('==========================================')
	print('NOTES: ')
	print(np.argmax(encoder_input_data[0],axis=-1))
	print("PREDICTED CHORDS : ")
	print(np.argmax(target_seq[0],axis=-1))
	print("REAL CHORDS")
	print(np.argmax(decoder_target_data[0],axis=-1))
	#input('next')
	print('==========================================')


if __name__ == "__main__":
	#create_model(256)
	train()
	#predict()