# -*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
import os
import pickle

# from tensorflow.nn.rnn_cell import GRUCell
from glove import *
from cnn import *
from model import Model

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
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['answer_as_number'], res['length']

vocab = []

def build_graph(vocab_size = len(vocab), state_size = 64, batch_size = 256, num_classes = 6):

    # reset_graph()

    # Placeholders
    ques_placeholder = tf.placeholder(tf.float32, [batch_size, None]) # [batch_size, num_steps]
    ques_seqlen = tf.placeholder(tf.int32, [batch_size])
    img_placeholder = tf.placeholder(tf.float32, [batch_size, None])
    ans_placeholder = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)

    # Embedding layer
    word_embeddings = tf.Variable(wordEmbeddings)
    rnn_word_inputs = tf.reshape(tf.nn.embedding_lookup(word_embeddings, ques_placeholder), [tf.shape(ques_placeholder)[0], self.max_length, self.config.n_features*self.config.embed_size])
    img_embeddings =  tf.Variable(imgEmbeddings)
    rnn_img_inputs = tf.reshape(tf.nn.embedding_lookup(img_embeddings, img_placeholder), [tf.shape(img_placeholder)[0], self.max_length, self.config.n_features*self.config.embed_size])

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size], initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_word_inputs, sequence_length=seqlen, initial_state=init_state)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))
    # idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
    # last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W_w = tf.get_variable('W_w', [state_size, num_classes])
        W_i = tf.get_variable('W_i', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(last_rnn_output, W_w) tf.matmul(img_placeholder, W_i)+ b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

# def train_graph(graph, batch_size = 256, num_epochs = 10, iterator = PaddedDataIterator):
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         tr = iterator(train)
#         te = iterator(test)

#         step, accuracy = 0, 0
#         tr_losses, te_losses = [], []
#         current_epoch = 0
#         while current_epoch < num_epochs:
#             step += 1
#             batch = tr.next_batch(batch_size)
#             feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
#             accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
#             accuracy += accuracy_

#             if tr.epochs > current_epoch:
#                 current_epoch += 1
#                 tr_losses.append(accuracy / step)
#                 step, accuracy = 0, 0

#                 #eval test set
#                 te_epoch = te.epochs
#                 while te.epochs == te_epoch:
#                     step += 1
#                     batch = te.next_batch(batch_size)
#                     feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
#                     accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
#                     accuracy += accuracy_

#                 te_losses.append(accuracy / step)
#                 step, accuracy = 0,0
#                 print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

#     return tr_losses, te_losses

########Read the data and create the data frame########

file_name = "data/data.pkl"
vocab_mapping = "data/vocab.pkl"
img_mapping = "data/img.pkl"
if(os.path.exists(file_name)):
	data_df = pd.read_pickle(file_name)
	words_df = pd.read_pickle(vocab_mapping)
	vocab = words_df.set_index(str('word'))["number"].to_dict()
	vocab = {str(k.encode('ascii')):vocab[k] for k in vocab}
	img_df = pd.read_pickle(img_mapping)
	img_vocab = img_df.set_index(str('img'))["number"].to_dict()
	img_vocab = {str(k.encode('ascii')):img_vocab[k] for k in img_vocab}
else:
	with open('data/question_answers.json') as data_file:
		data = json.load(data_file)
	sen_length = collections.defaultdict(int)
	num_single_word_ans = 0
	vocab = {}
	img_vocab = {}
	word_count = 0
	img_count = 0
	np_data = np.array([['Index','image_id','question', 'as_numbers', 'length', 'answer', 'answer_as_number']])
	for indx in range(len(data)):
		img_iter = data[indx]
		img = str(img_iter["image_id"])
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
				np_data = np.append(np_data, [["Row"+str(num_single_word_ans), img_iter, ques["question"], ques_numbers, len(ques_string), answer, vocab[answer]]], axis=0)
				if(np_data.shape[0]%10000 == 0):
					print np_data.shape
		if(np_data.shape[0]>10000):
			break
	data_df = pd.DataFrame(data=np_data[1:,1:], index=np_data[1:,0], columns=np_data[0,1:])
	data_df.to_pickle(file_name)
	words_df = pd.DataFrame(data=vocab.items(), columns=['word', 'number'])
	words_df.to_pickle(vocab_mapping)

print data_df.shape

# g = build_graph()
# tr_losses, te_losses = train_graph(g)
wordEmbeddings = loadWordVectors(vocab)
# imgEmbeddings = loadImgVectors(img_vocab)

