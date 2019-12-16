from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from custom_recurrents import AttentionDecoder
import numpy as np
import sys
import time

def train(attention):
	max_len = 2286
	batch_size = 64
	epochs = 20
	latent_dim = 256
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(suppress=True)
	path = '../../../../data/data1/users/dsyrrafos/seq2seq/'


	encoder_input_data = np.load(path+'encoder_input_data.npy')
	decoder_input_data = np.load(path+'decoder_input_data.npy')
	decoder_target_data = np.load(path+'decoder_target_data.npy')
	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, 14))
	encoder = LSTM(latent_dim, return_state = True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]


	if not attention:
		# Set up the decoder, using `encoder_states` as initial state.
		decoder_inputs = Input(shape=(None, 38))
		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the
		# return states in the training model, but we will use them in inference.
		decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)

		decoder_dense = Dense(38,activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)
	else:
		decoder_inputs = Input(shape=(None, 38))
		decoder_outputs = AttentionDecoder(decoder_inputs)

	'''
	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.summary()
	# Run training
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	tensorboard = TensorBoard(log_dir=path+"logs/{}".format(time.time())+'_'+str(epochs))
	model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[tensorboard])

	# Save model
	model_path = model_path = '../../../../data/data1/users/dsyrrafos/models/'
	model.save(model_path+'s2s_ld256_ep20.h5')
	'''
	# Next: inference mode (sampling).
	# Here's the drill:
	# 1) encode input and retrieve initial decoder state
	# 2) run one step of decoder with this initial state
	# and a "start of sequence" token as target.
	# Output will be the next target token
	# 3) Repeat with the current target token and current states

	# Define sampling models
	model_path = model_path = '../../../../data/data1/users/dsyrrafos/models/'
	model = load_model(model_path+'s2s_ld256_ep20.h5')
	model.summary()
	plot_model(model, to_file=path+'model.png')
	print(len(model.layers))
	for i in range(len(model.layers)):
		print(i)
		print(model.layers[i])
	encoder_model = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
	encoder_model.summary()
	plot_model(encoder_model, to_file=path+'encoder_model.png')
	'''
	decoder_input = Input(shape=(latent_dim,))
	decoder_model = decoder_input
	_, decoder_state_input_h, decoder_state_input_c = model.layers[2].output
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	_, state_h, state_c = model.layers[3].output
	decoder_states = [state_h, state_c]
	'''
	for i in range(len(model.layers)):
		print(i, model.layers[i])
	#decoder_input = Input(model.layers[1].input_shape[1:])	
	decoder_input = model.layers[1].input, Input(shape=(latent_dim,)), Input(shape=(latent_dim,))
	decoder_states_input = [decoder_input[1], decoder_input[2]]
	decoder_model = decoder_input
	#decoder_model = model.layers[3](decoder_model)
	lstm_out, lstm_h, lstm_c = model.layers[3](decoder_model[0], initial_state=decoder_states_input)
	decoder_states = [lstm_h, lstm_c]
	decoder_model = model.layers[4](lstm_out)
	decoder_model = Model(inputs=decoder_input, outputs=[decoder_model] + decoder_states)
	decoder_model.summary()
	plot_model(decoder_model, to_file=path+'decoder_model.png')

	'''
	#encoder_model = Model(encoder_inputs, encoder_states)
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(
	    decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model(
	    [decoder_inputs] + decoder_states_inputs,
	    [decoder_outputs] + decoder_states)
	'''

	
	def decode_sequence(input_seq):
		# Encode the input as state vectors.
		#states_value = encoder_model.predict(input_seq)
		enc_out, enc_h, enc_c = encoder_model.predict(input_seq)
		states_value = [enc_h, enc_c]
		print ('--------states_value----------')
		print(states_value)

		# Generate empty target sequence of length 1.
		target_seq = np.zeros((1, 1, 38))
		target_seq[0,0,36] = 1

		# Sampling loop for a batch of sequences
		# (to simplify, here we assume a batch of size 1).
		stop_condition = False
		decoded_seq = np.empty((1,1,38))
		counter=0
		while not stop_condition:
			#print(counter)
			output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
			#print('output_tokens.shape', output_tokens.shape)
			#print(output_tokens)
			#decoded_seq = np.append(decoded_seq, output_chord)
			#decoded_seq = np.concatenate((decoded_seq, output_chord), axis=1)

			# Sample a token
			sampled_token_index = np.argmax(output_tokens[0, -1, :])
			#print('sampled_token_index.shape', sampled_token_index.shape)
			#print('sampled_token_index', sampled_token_index)

			output_chord = np.zeros((1,1,38))
			output_chord[0,0,sampled_token_index] = 1
			#print(output_chord)
			decoded_seq = np.concatenate((decoded_seq, output_chord), axis=1)
			
			# Exit condition: either hit max length
			# or find stop character.
			if (sampled_token_index == 37 or np.size(decoded_seq,1) == max_len):
				stop_condition = True
				
			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1, 38))
			target_seq[0, 0, sampled_token_index] = 1.

			# Update states
			states_value = [h, c]
			counter +=1
			#print(states_value)

		return decoded_seq

	for seq_index in range(105,108):
		# Take one sequence (part of the training set)
		# for trying out decoding.
		input_seq = encoder_input_data[seq_index: seq_index + 1]
		decoded_seq = decode_sequence(input_seq)
		print('-')
		#print('Input sentence:', encoder_input_data[seq_index])
		#print('Decoded sentence:', decoded_seq)
		print('input_seq.shape', input_seq.shape)
		print('decoded_seq.shape', decoded_seq.shape)
		print(input_seq[:,:128,:])
		print(decoded_seq[:,:128,:])

	return 0

def sample():	


		return 0


def evaluate():
	path = '../../../../data/data1/users/dsyrrafos/'

	batch_size = 64
	model = load_model('s2s_20_fix.h5')
	test_encoder_input_data = np.load(path+'test_encoder_input_data.npy')
	test_decoder_input_data = np.load(path+'test_decoder_input_data.npy')
	test_decoder_target_data = np.load(path+'test_decoder_target_data.npy')

	tensorboard = TensorBoard(log_dir=path+"logs/{}".format(time.time())+'_eval')
	results = model.evaluate([test_encoder_input_data, test_decoder_input_data], test_decoder_target_data, batch_size=batch_size, callbacks=[tensorboard])
	print('test loss, test acc:', results)

	return 0


def vectorize_data():
	path = '../../../../data/data1/users/dsyrrafos/'

	#Vectorize the data.
	notes = np.load(path+'notes_final.npy', encoding='latin1')
	chords = np.load(path+'sparse_final.npy', encoding='latin1')
	print('input_melody shape: ', len(notes))
	print('target_chords shape: ', len(chords))

	path = '../../../../data/data1/users/dsyrrafos/seq2seq/'

	number_of_tracks = len(notes)
	print('number_of_tracks:', number_of_tracks)

	max_len = 0
	for i in range(0, number_of_tracks):
		if np.size(notes[i], 0) > max_len:
			max_len = np.size(notes[i], 0)
	print('max_len:', max_len)

	notes_train, notes_test, chords_train, chords_test = train_test_split(notes, chords, test_size = 0.2, random_state = 1)

	notes = notes_train
	chords = chords_train

	number_of_tracks = len(notes)
	print('number_of_train_tracks:', number_of_tracks)

	print('adding zeros ...')


	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		while np.size(notes[i], 0) < max_len:
			notes[i] = np.append(notes[i], np.zeros((1,14)), 0)
			chords[i] = np.append(chords[i], np.zeros((1,38)), 0)


	#for i in range(0, number_of_tracks):
	#	print(i, np.size(notes[i], 0))

	encoder_input_data = notes[0]
	encoder_input_data = np.array(encoder_input_data)
	encoder_input_data = np.expand_dims(encoder_input_data, axis=0)
	decoder_input_data = chords[0]
	decoder_input_data = np.array(decoder_input_data)
	decoder_input_data = np.expand_dims(decoder_input_data, axis=0)
	decoder_target_data = np.zeros((number_of_tracks, max_len, 38))
	print('encoder_input_data.shape: '+str(encoder_input_data.shape))

	print('expanding dimensions...')
	for i in range(1, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		notes[i] = np.expand_dims(notes[i], axis=0)
		chords[i] = np.expand_dims(chords[i], axis=0)

	print('stacking pianorolls...')
	for i in range(1, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		encoder_input_data = np.concatenate((encoder_input_data, notes[i]), axis=0)
		decoder_input_data = np.concatenate((decoder_input_data, chords[i]), axis=0)

	print('shape input_melody stacked', encoder_input_data.shape)
	print('shape target_chords stacked', decoder_input_data.shape)

	print('creating decoder_target_data...')
	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		for t in range(0, max_len-1):
			for j in range(0, 38):
				decoder_target_data[i, t, j] = decoder_input_data[i, t+1, j]

	np.save(path+'encoder_input_data.npy', encoder_input_data)
	np.save(path+'decoder_input_data.npy', decoder_input_data)
	np.save(path+'decoder_target_data.npy', decoder_target_data)


	# Vectorize test data
	notes = notes_test
	chords = chords_test

	number_of_tracks = len(notes)
	print('number_of_train_tracks:', number_of_tracks)

	print('adding zeros ...')
	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		while np.size(notes[i], 0) < max_len:
			notes[i] = np.append(notes[i], np.zeros((1,14)), 0)
			chords[i] = np.append(chords[i], np.zeros((1,38)), 0)

	#for i in range(0, number_of_tracks):
	#	print(i, np.size(notes[i], 0))

	test_encoder_input_data = notes[0]
	test_encoder_input_data = np.array(test_encoder_input_data)
	test_encoder_input_data = np.expand_dims(test_encoder_input_data, axis=0)
	test_decoder_input_data = chords[0]
	test_decoder_input_data = np.array(test_decoder_input_data)
	test_decoder_input_data = np.expand_dims(test_decoder_input_data, axis=0)
	test_decoder_target_data = np.zeros((number_of_tracks, max_len, 38))


	print('test_encoder_input_data.shape: '+str(test_encoder_input_data.shape))

	print('expanding dimensions...')
	for i in range(1, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		notes[i] = np.expand_dims(notes[i], axis=0)
		chords[i] = np.expand_dims(chords[i], axis=0)

	print('stacking pianorolls...')
	for i in range(1, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		test_encoder_input_data = np.concatenate((test_encoder_input_data, notes[i]), axis=0)
		test_decoder_input_data = np.concatenate((test_decoder_input_data, chords[i]), axis=0)

	print('shape input_melody stacked', test_encoder_input_data.shape)
	print('shape target_chords stacked', test_decoder_input_data.shape)

	print('creating decoder_target_data...')
	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		for t in range(0, max_len-1):
			for j in range(0, 38):
				test_decoder_target_data[i, t, j] = test_decoder_input_data[i, t+1, j]

	np.save(path+'test_encoder_input_data.npy', test_encoder_input_data)
	np.save(path+'test_decoder_input_data.npy', test_decoder_input_data)
	np.save(path+'test_decoder_target_data.npy', test_decoder_target_data)

	return 0


def small_arrays():
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(suppress=True)
	path = '../../../../data/data1/users/dsyrrafos/'

	#Vectorize the data.
	notes = np.load(path+'notes_final.npy', encoding='latin1')
	chords = np.load(path+'sparse_final.npy', encoding='latin1')
	print('input_melody shape: ', notes.shape)
	print('target_chords shape: ', chords.shape)

	path = '../../../../data/data1/users/dsyrrafos/4bars/'

	number_of_tracks = len(notes)
	print('number_of_tracks:', number_of_tracks)

	max_len = 66

	print('splitting arrays ...')

	# start and stop characters for splitting arrays
	start_char_notes = np.zeros((1,14))
	start_char_notes[0,12] = 1
	stop_char_notes = np.zeros((1,14))
	stop_char_notes[0,13] = 1
	start_char_chords = np.zeros((1,38))
	start_char_chords[0,36] = 1
	stop_char_chords = np.zeros((1,38))
	stop_char_chords[0,37] = 1

	stack_notes = np.empty((1,66,14))
	stack_chords = np.empty((1,66,38))

	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		#print('notes[i].shape', notes[i].shape)
		count = 0
		while np.size(notes[i], 0) > 66:
			#------------------------------Notes------------------------------
			new_array, notes[i] = np.vsplit(notes[i], np.array([65]))
			#print(new_array.shape)
			#print(notes[i].shape)
			new_array = np.append(new_array, stop_char_notes, axis=0)
			notes[i] = np.insert(notes[i], 0, start_char_notes, axis=0)
			#print("new_array.shape", new_array.shape)
			#print("notes[i].shape", notes[i].shape)
			#print('after slice', len(notes[i]))
			new_array = np.expand_dims(new_array, axis=0)
			stack_notes = np.concatenate((stack_notes, new_array), axis=0)
			#------------------------------Chords------------------------------
			new_array, chords[i] = np.vsplit(chords[i], np.array([65]))
			#print(new_array.shape)
			#print(notes[i].shape)
			new_array = np.append(new_array, stop_char_chords, axis=0)
			chords[i] = np.insert(chords[i], 0, start_char_chords, axis=0)
			#print("new_array.shape", new_array.shape)
			#print("notes[i].shape", notes[i].shape)
			#print('after slice', len(notes[i]))
			new_array = np.expand_dims(new_array, axis=0)
			stack_chords = np.concatenate((stack_chords, new_array), axis=0)
			count +=1
		while np.size(notes[i], 0) < 66:
			notes[i] = np.append(notes[i], np.zeros((1,14)), 0)
			chords[i] = np.append(chords[i], np.zeros((1,38)), 0)
		notes[i] = np.expand_dims(notes[i], axis=0)
		chords[i] = np.expand_dims(chords[i], axis=0)
		stack_notes = np.concatenate((stack_notes, notes[i]), axis=0)
		stack_chords = np.concatenate((stack_chords, chords[i]), axis=0)
		#print('splitted '+str(count)+' times')


	stack_notes = np.delete(stack_notes, 0, axis=0)
	stack_chords = np.delete(stack_chords, 0, axis=0)
	print(stack_notes.shape)
	print(stack_chords.shape)

	print('creating decoder_target_data...')
	for i in range(0, number_of_tracks):
		print(str(i)+'/'+str(number_of_tracks))
		for t in range(0, max_len-1):
			for j in range(0, 38):
				stack_targets[i, t, j] = stack_chords[i, t+1, j]

	np.save(path+'test_encoder_input_data.npy', stack_notes)
	np.save(path+'test_decoder_input_data.npy', stack_chords)
	np.save(path+'test_decoder_target_data.npy', stack_targets)


	return 0


if __name__ == "__main__":
	train(0)
