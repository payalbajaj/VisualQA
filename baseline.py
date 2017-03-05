# -*- coding: utf-8 -*-

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
file_name = "../data_VisualQA/data.pkl"
vocab_mapping = "../data_VisualQA/vocab.pkl"
img_mapping = "../data_VisualQA/img.pkl"
if(os.path.exists(file_name) and os.path.exists(vocab_mapping) and os.path.exists(img_mapping)):
	data_df = pd.read_pickle(file_name)
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
	np_data = [('image_id', 'image_as_number', 'question', 'as_numbers', 'length', 'answer', 'answer_as_number')]
	for indx in range(len(data)):
		img_iter = data[indx]
		img = str(img_iter["id"])
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
				if(len(np_data)%10000 == 0):
					print len(np_data)
	data_df = pd.DataFrame(data=np_data[1:], columns=list(np_data[0]))
	data_df.to_pickle(file_name)
	words_df = pd.DataFrame(data=vocab.items(), columns=['word', 'number'])
	words_df.to_pickle(vocab_mapping)
	image_df = pd.DataFrame(data=img_vocab.items(), columns=['img', 'number'])
	image_df.to_pickle(img_mapping)

wordEmbeddings = loadWordVectors(vocab) #check if you need embedding for "UNK"
imgEmbeddings = loadImgVectors(img_vocab)


#########Building the Baseline Graph############

ques_embed_size = 50    #golve vectors are 50 dimensional
img_embed_size = 50 #replace this by size of image embeddings
hidden_state_size = ques_embed_size     #can be changed
batch_size = 100


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
    keep_prob = tf.constant(1.0)

    # Embedding layer
    word_embeddings = tf.Variable(wordEmbeddings, dtype=tf.float32)
    rnn_word_inputs = tf.nn.embedding_lookup(word_embeddings, ques_placeholder)
    # rnn_word_inputs = tf.reshape(tf.nn.embedding_lookup(word_embeddings, ques_placeholder), [tf.shape(ques_placeholder)[0], max_ques_length, ques_embed_size])
    img_embeddings =  tf.Variable(imgEmbeddings, dtype=tf.float32)
    rnn_img_inputs = tf.nn.embedding_lookup(img_embeddings, img_placeholder)
    # rnn_img_inputs = tf.reshape(tf.nn.embedding_lookup(img_embeddings, img_placeholder), [tf.shape(img_placeholder)[0], img_embed_size])

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(ques_embed_size, hidden_state_size)
    init_state = tf.get_variable('init_state', [1, hidden_state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_word_inputs, sequence_length=ques_seqlen_placeholder, initial_state=init_state, dtype=tf.float32)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    # last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), ques_seqlen_placeholder-1], axis=1))
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (ques_seqlen_placeholder - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, hidden_state_size]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W_ques = tf.get_variable('W_ques', [hidden_state_size, num_classes], dtype=tf.float32)
        W_img = tf.get_variable('W_img', [img_embed_size, num_classes], dtype=tf.float32)
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    
    logits = tf.matmul(last_rnn_output, W_ques) + tf.matmul(rnn_img_inputs, W_img) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), ans_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ans_placeholder))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'ques_placeholder': ques_placeholder,
        'ques_seqlen_placeholder': ques_seqlen_placeholder,
        'img_placeholder': img_placeholder,
        'ans_placeholder': ans_placeholder,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

def train_graph(graph, batch_size = batch_size, num_epochs = 100, iterator = PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tr = iterator(data_df)
        # te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['ques_placeholder']: batch[0], g['img_placeholder']: batch[1], g['ans_placeholder']: batch[2], g['ques_seqlen_placeholder']: batch[3]}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                #eval test set
                # te_epoch = te.epochs
                # while te.epochs == te_epoch:
                #     step += 1
                #     batch = te.next_batch(batch_size)
                #     feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                #     accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                #     accuracy += accuracy_

                # te_losses.append(accuracy / step)
                # step, accuracy = 0,0
                # print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1])

    return tr_losses, te_losses

g = build_graph(batch_size=batch_size)
tr_losses, te_losses = train_graph(g)