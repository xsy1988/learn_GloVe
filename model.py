# Create by MrZhang on 2019-11-28

import numpy as np

class glove_model(object):

    def __init__(self, input_matrix, vocab_list, epoch, learn_rate, emb_save_path, loss_save_path):
        self.input_matrix = input_matrix
        self.vocab_list = vocab_list
        self.epoch = epoch
        self.learn_rate = learn_rate
        self.emb_save_path = emb_save_path
        self.loss_save_path = loss_save_path
        self.dim = 100
        self.x_max = 30

    def create_weights(self, vocab_len):
        emb_matrix = np.random.randn(self.dim, vocab_len)
        bias_array = np.random.randn(vocab_len)
        return emb_matrix, bias_array

    def weight_function(self, x):
        if x < self.x_max:
            return np.power((x / self.x_max), 0.75)
        else:
            return 1

    def forward_propagate(self, emb_matrix, bias_array, index_i, index_k):
        w_i = emb_matrix[:, index_i]
        w_k = emb_matrix[:, index_k]
        b_i = bias_array[index_i]
        b_k = bias_array[index_k]
        x_ik = self.input_matrix[index_i][index_k] + 0.00001
        diff = np.dot(w_i.T, w_k) + b_i + b_k - np.log(x_ik)
        loss = 0.5 * self.weight_function(x_ik) * diff * diff
        calc_param = {"w_i": w_i, "w_k": w_k, "x_ik": x_ik}
        return loss, diff, calc_param

    def backward_propagate(self, diff, calc_param):
        w_i = calc_param["w_i"]
        w_k = calc_param["w_k"]
        x_ik = calc_param["x_ik"]
        dw_i = self.weight_function(x_ik) * diff * w_k
        dw_k = self.weight_function(x_ik) * diff * w_i
        db_i = db_k = self.weight_function(x_ik) * diff
        grads = {"dw_i": dw_i, "dw_k": dw_k, "db_i": db_i, "db_k": db_k}
        return grads

    def one_circle(self, emb_matrix, bias_array, index_i, index_k):
        loss, diff, calc_param = self.forward_propagate(emb_matrix, bias_array, index_i, index_k)
        grads = self.backward_propagate(diff, calc_param)
        dw_i = grads["dw_i"]
        dw_k = grads["dw_k"]
        db_i = grads["db_i"]
        db_k = grads["db_k"]
        emb_matrix[:, index_i] -= self.learn_rate * dw_i
        emb_matrix[:, index_k] -= self.learn_rate * dw_k
        bias_array[index_i] -= self.learn_rate * db_i
        bias_array[index_k] -= self.learn_rate * db_k
        return loss

    def train(self):
        vocab_len = len(self.vocab_list)
        emb_matrix, bias_array = self.create_weights(vocab_len)
        loss_list = []
        for e in range(self.epoch):
            loss = 0.0
            for i in range(vocab_len):
                for j in range(vocab_len):
                    loss += self.one_circle(emb_matrix, bias_array, i, j)
            loss_list.append(loss)
            print("in epoch {}, loss is {}".format(e, loss))
        loss_tag = np.array(loss_list)
        np.save(self.loss_save_path, loss_tag)
        np.save(self.emb_save_path, emb_matrix)
