# Create by MrZhang on 2019-11-28

import numpy as np
import model
import matplotlib.pyplot as plt

input_matrix = np.load('datas/co_mat.npy')
vocab_list = np.load('datas/vocab_list.npy')
loss_save_path = 'datas/loss.npy'
emb_save_path = 'datas/embedding.npy'
epoch = 200
learn_rate = 0.01

glove_model = model.glove_model(input_matrix, vocab_list, epoch, learn_rate, emb_save_path, loss_save_path)

glove_model.train()

loss = np.load(loss_save_path)
plt.plot(loss, '.-', color='r' ,label="loss")
plt.show()
