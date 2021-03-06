#-*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
import os
import pickle
from glove import *
from cnn import *
from model import Model
from nltk.corpus import wordnet

max_ques_length = 24

class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        pass

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        # maxlen = max(res['length'])
        x = np.zeros([n, max_ques_length], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['image_as_number'], res['answer_as_number'], res['length']

########Read the data and create the data frame########
data_path = "../data_VisualQA/question_answers.json"
file_name = "../data_VisualQA/data_trainFinal.pkl"
file_name2 = "../data_VisualQA/data_devFinal.pkl"
file_name3 = "../data_VisualQA/data_testFinal.pkl"
vocab_mapping = "../data_VisualQA/vocabFinal.pkl"
img_mapping = "../data_VisualQA/imgFinal.pkl"
if(os.path.exists(file_name) and os.path.exists(file_name2) and os.path.exists(file_name3) and os.path.exists(vocab_mapping) and os.path.exists(img_mapping)):
	data_train = pd.read_pickle(file_name)
	data_dev = pd.read_pickle(file_name2)
	data_test = pd.read_pickle(file_name3)
	words_df = pd.read_pickle(vocab_mapping)
    #create dictionary needed for embeddings
	vocab = words_df.set_index(str('word'))["number"].to_dict()
	vocab = {str(k.encode('ascii')):vocab[k] for k in vocab}
	img_df = pd.read_pickle(img_mapping)
    #create dictionary needed for embeddings
	img_vocab = img_df.set_index(str('img'))["number"].to_dict()
	img_vocab = {str(k.encode('ascii')):img_vocab[k] for k in img_vocab}
else:
	with open(data_path) as data_file:
		data = json.load(data_file)
	sen_length = collections.defaultdict(int)
	num_single_word_ans = 0
	vocab = {}
	img_vocab = {}
	word_count = 0
	vocab["UNK"] = word_count
	word_count += 1
	img_count = 0
	###Read images for which embeddings are present and create data frame only for those images####
	img_set = set()
	f = open("../data_VisualQA/cnn1500.txt", "r")
	for line in f:
		img_id = line.split(" ")[0].split('/')[4].split('.')[0]
		img_set.add(img_id)
	print len(img_set)
	f.close()
	np_data = [('image_id', 'image_as_number', 'question', 'as_numbers', 'length', 'answer', 'answer_as_number')]
	for indx in range(len(data)):
		img_iter = data[indx]
		img = str(img_iter["id"])
		###Read images for which embeddings are present and create data frame only for those images####
		if(img not in img_set):
			continue
		if(img not in img_vocab):
			img_vocab[img] = img_count
			img_count += 1
		for ques in img_iter["qas"]:
			if(" " not in ques["answer"]):	#single word answer
				answer = ques["answer"].replace(".","")
				num_single_word_ans += 1
				ques_string = ques["question"].replace("?","").lower().split(" ")
				sen_length[len(ques_string)] += 1
				for word in ques_string:
					if word not in vocab:
						vocab[word] = word_count
						word_count += 1
				if answer not in vocab:
					vocab[answer] = word_count
					word_count += 1
				ques_numbers = [vocab[word] for word in ques_string]
				np_data.append((img, img_vocab[img], ques["question"], ques_numbers, len(ques_string), answer, vocab[answer]))
				if(len(np_data)%100 == 0):
					print len(np_data)
	data_df = pd.DataFrame(data=np_data[1:], columns=list(np_data[0]))
	#Split data - 70% Train, 20% Dev, 10% Test
	#Split into train and remaining
	msk = np.random.rand(len(data_df)) < 0.7
	data_train = data_df[msk]
	data_rem = data_df[~msk]
	#Split the remaining data into dev and test
	msk = np.random.rand(len(data_rem)) < 0.67
	data_dev = data_rem[msk]
	data_test = data_rem[~msk]
	data_train.to_pickle(file_name)
	data_dev.to_pickle(file_name2)
	data_test.to_pickle(file_name3)
	words_df = pd.DataFrame(data=vocab.items(), columns=['word', 'number'])
	words_df.to_pickle(vocab_mapping)
	image_df = pd.DataFrame(data=img_vocab.items(), columns=['img', 'number'])
	image_df.to_pickle(img_mapping)

wordEmbeddings = loadWordVectors(vocab) #check if you need embedding for "UNK"
imgEmbeddings = loadImgVectors(img_vocab, filepath = "../data_VisualQA/cnn1500.txt")


#########Building the Baseline Graph############

ques_embed_size = 200    #glove vectors are 50 dimensional
img_embed_size = 512 #replace this by size of image embeddings
hidden_state_size = ques_embed_size    #can be changed
batch_size = 128
N = 196
T = 1

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(batch_size, num_classes=len(vocab)):    #num_classes should be equal to len(vocab)

    reset_graph()

    # Placeholders
    ques_placeholder = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    ques_seqlen_placeholder = tf.placeholder(tf.int32, [batch_size])
    img_placeholder = tf.placeholder(tf.int32, [batch_size])	
    ans_placeholder = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(0.7)

    # Embedding layer
    word_embeddings = tf.Variable(wordEmbeddings, dtype=tf.float32)
    rnn_word_inputs = tf.nn.embedding_lookup(word_embeddings, ques_placeholder)
    img_embeddings =  tf.Variable(imgEmbeddings, dtype=tf.float32)
    rnn_img_inputs = tf.nn.embedding_lookup(img_embeddings, img_placeholder)	#Assume that this will be (512 * 196 - that is 196 512-dimensional vectors)

	#Question Input Module
    with tf.variable_scope('wordGRU'):
    	cell = tf.contrib.rnn.GRUCell(ques_embed_size, hidden_state_size)
    	#cell = tf.nn.rnn_cell.GRUCell(ques_embed_size, hidden_state_size)
    init_state = tf.get_variable('init_state', [1, hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_word_inputs, sequence_length=ques_seqlen_placeholder, initial_state=init_state, dtype=tf.float32)
	# Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (ques_seqlen_placeholder - 1)
    ques_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, hidden_state_size]), idx)

    #Image Input Module
    #Reshape to 196 by 512
    rnn_img_inputs = tf.reshape(rnn_img_inputs, [batch_size, N, img_embed_size])
    #tanh mapping
    W_img_input = tf.get_variable('W_img_input', [img_embed_size, hidden_state_size], dtype=tf.float32)
    b_img_input = tf.get_variable('b_img_input', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    rnn_img_mapped = tf.tanh(tf.einsum('ijk,kl->ijl',rnn_img_inputs, W_img_input) + b_img_input)
    #Bi-directional GRU
    #Forward
    with tf.variable_scope('forward'):
	    cell_img_fwd = tf.contrib.rnn.GRUCell(hidden_state_size, hidden_state_size)
	    #cell_img_fwd = tf.nn.rnn_cell.GRUCell(hidden_state_size, hidden_state_size)
	    img_init_state_fwd = rnn_img_mapped[:, 0, :]
	    img_init_state_fwd = tf.multiply(img_init_state_fwd, tf.zeros([batch_size, hidden_state_size]))
	    rnn_outputs_fwd, final_state_fwd = tf.nn.dynamic_rnn(cell_img_fwd, rnn_img_mapped, initial_state=img_init_state_fwd, dtype=tf.float32)
    #Backward
    rnn_img_mapped_rev = tf.reverse(rnn_img_mapped, [1])
    #rnn_img_mapped_rev = tf.reverse(rnn_img_mapped, [False, True, False])
    with tf.variable_scope('backward'):
    	cell_img_bwd = tf.contrib.rnn.GRUCell(hidden_state_size, hidden_state_size)
    	#cell_img_bwd = tf.nn.rnn_cell.GRUCell(hidden_state_size, hidden_state_size)
    	img_init_state_bwd = tf.get_variable('img_init_state_bwd', [1, hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    	img_init_state_bwd = tf.tile(img_init_state_bwd, [batch_size, 1])
    	rnn_outputs_bwd, final_state_bwd = tf.nn.dynamic_rnn(cell_img_bwd, rnn_img_mapped_rev, initial_state=img_init_state_fwd, dtype=tf.float32)
    #Sum up the learned vectors to get facts of the image
    img_features = rnn_outputs_fwd+rnn_outputs_bwd

    #T rounds for updating Memory
    #Initialize Variables
    W_inner = tf.get_variable('W_inner', [4*hidden_state_size, hidden_state_size], dtype=tf.float32)
    W_outer = tf.get_variable('W_outer', [hidden_state_size, hidden_state_size], dtype=tf.float32)
    b_inner = tf.get_variable('b_inner', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    b_outer = tf.get_variable('b_outer', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    W_r = tf.get_variable('W_r', [hidden_state_size, hidden_state_size], dtype=tf.float32)
    U_r = tf.get_variable('U_r', [hidden_state_size, hidden_state_size], dtype=tf.float32)
    b_r = tf.get_variable('b_r', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    W_h = tf.get_variable('W_h', [hidden_state_size, hidden_state_size], dtype=tf.float32)
    U_h = tf.get_variable('U_h', [hidden_state_size, hidden_state_size], dtype=tf.float32)
    b_h = tf.get_variable('b_h', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    W_t = tf.get_variable('W_t', [3*hidden_state_size, hidden_state_size], dtype=tf.float32)
    b_t = tf.get_variable('b_t', [hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    W_a = tf.get_variable('W_a', [hidden_state_size, num_classes], dtype=tf.float32)
    #Initialize memory to questiosn 
    m = ques_rnn_output
    #Initialize context
    c = tf.zeros((1,hidden_state_size))
    for t in range(T):
    	print t
    	prev_m = m
    	h = c
    	# Attention Gates
    	for i in range(N):
	    	z = tf.concat([tf.multiply(img_features[:,i,:], ques_rnn_output), tf.multiply(img_features[:,i,:], prev_m),tf.abs(img_features[:,i,:]-ques_rnn_output),tf.abs(img_features[:,i,:]-prev_m)], 1)
	    	#z = tf.concat(1,[tf.multiply(img_features[:,i,:], ques_rnn_output), tf.multiply(img_features[:,i,:], prev_m),tf.abs(img_features[:,i,:]-ques_rnn_output),tf.abs(img_features[:,i,:]-prev_m)])
	    	Z = tf.matmul(tf.tanh(tf.matmul(z,W_inner)+b_inner),W_outer)+b_outer
	    	g = tf.nn.softmax(Z)
		    # Attention Mechanism - Attention based GRU
	    	r = tf.nn.sigmoid(tf.matmul(img_features[:,i,:],W_r) + tf.matmul(h,U_r) + b_r)
	    	hprime = tf.tanh(tf.matmul(img_features[:,i,:],W_h) + tf.multiply(r,tf.matmul(h,U_h)) + b_h)
	    	h=tf.multiply(g,hprime)+tf.multiply((1-g),h)
	    #Update context
		c = h
	    # Memory Update using the final state of the Attention based GRU
		#m = tf.nn.relu(tf.matmul(tf.concat(1, [prev_m,c,ques_rnn_output]),W_t) + b_t)
		m = tf.nn.relu(tf.matmul(tf.concat([prev_m,c,ques_rnn_output], 1),W_t) + b_t)


	logits = tf.matmul(m,W_a)
	preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), ans_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ans_placeholder))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    preds_idx = tf.argmax(preds, 1)


    return {
        'ques_placeholder': ques_placeholder,
        'ques_seqlen_placeholder': ques_seqlen_placeholder,
        'img_placeholder': img_placeholder,
        'ans_placeholder': ans_placeholder,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'preds_idx' : preds_idx,
        'accuracy': accuracy
    }

def train_graph(g, batch_size = batch_size, num_epochs = 256, iterator = PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.initialize_all_variables())
        #Train Data
        tr = iterator(data_train)
        #Dev Data
        dev = iterator(data_dev)
        #Test Data
        te = iterator(data_test)

        step, accuracy = 0, 0
        tr_losses, dev_losses, te_losses = [], [], []
	wupsscores = []
	current_epoch = 0
	curwups = []
	predact = []
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['ques_placeholder']: batch[0], g['img_placeholder']: batch[1], g['ans_placeholder']: batch[2], g['ques_seqlen_placeholder']: batch[3]}
            accuracy_, preds_idx_, _ = sess.run([g['accuracy'], g['preds_idx'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            idx = 0
            
	    for ent in batch[2]:
            	pred_idx = preds_idx_[idx]
            	actual_idx = ent
            	idx += 1
            	for word, word_idx in vocab.iteritems():
            		if word_idx == pred_idx:
            			pred_word = word
            		if word_idx == actual_idx:
            			actual_word = word
            	wordFromList1 = wordnet.synsets(pred_word)
            	wordFromList2 = wordnet.synsets(actual_word)
		if wordFromList1 and wordFromList2:
			predact.append('predicted: ' + pred_word + ' , actual: ' + actual_word)
            		s = wordFromList1[0].wup_similarity(wordFromList2[0])
            		if(s != None):
            			curwups.append(s)
	    if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0
            	avgscore = sum(curwups)/len(curwups)
		wupsscores.append(avgscore)
		curwups =  []
                # eval dev set
                dev_epoch = dev.epochs
                while dev.epochs == dev_epoch:
                    step += 1
                    batch = dev.next_batch(batch_size)
                    feed = {g['ques_placeholder']: batch[0], g['img_placeholder']: batch[1], g['ans_placeholder']: batch[2], g['ques_seqlen_placeholder']: batch[3]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                dev_losses.append(accuracy / step)
                step, accuracy = 0,0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- dev:", dev_losses[-1])
		if current_epoch != 256:
			del predact[:]

        #Run on test data and get accuracy here
        te_epoch = te.epochs
        while te.epochs == te_epoch:
            step += 1
            batch = te.next_batch(batch_size)
            feed = {g['ques_placeholder']: batch[0], g['img_placeholder']: batch[1], g['ans_placeholder']: batch[2], g['ques_seqlen_placeholder']: batch[3]}
            accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
            accuracy += accuracy_
        te_losses.append(accuracy / step)
        step, accuracy = 0,0
    
    return tr_losses, dev_losses, te_losses, wupsscores, predact

g = build_graph(batch_size=batch_size)
tr_losses, dev_losses, te_losses, avg_scores, predact = train_graph(g)
np.savetxt('results/Final/trainingloss.txt', np.array(tr_losses), delimiter='\n')
np.savetxt('results/Final/devloss.txt', np.array(dev_losses), delimiter='\n')
np.savetxt('results/Final/testloss.txt',np.array(te_losses), delimiter='\n')
np.savetxt('results/Final/avgWUPSscores.txt',np.array(avg_scores), delimiter='\n')
np.savetxt('results/Final/erroranalysis.txt',np.array(predact), delimiter='\n', fmt = '%s')
