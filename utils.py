# Create by MrZhang on 2019-11-27

import numpy as np
import re

class create_co_occurrence(object):

    def __init__(self, native_data_path, vocab_save_path, mat_save_path, window_size):
        self.native_data_path = native_data_path
        self.vocab_save_path = vocab_save_path
        self.mat_save_path = mat_save_path
        self.window_size = window_size

    # 对原始数据做处理，去掉各类符号，然后按句子进行切分，使数据集中每一行是一个单独的句子。
    def clear_data(self):
        fr = open(self.native_data_path, 'r')
        sentence_list = []
        for line in fr:
            line = line.strip('\n')
            line = re.sub(r'[{}]+'.format('“”!,;:?"-_'), '', line)
            line = re.sub(r'\d+', '', line)
            line = re.sub(r'[{}]+'.format('\'s'), '', line)
            sentences = line.split('.')
            sent_list = [sentence for sentence in sentences if sentence != '']
            for sentence in sent_list:
                sentence_list.append(sentence.strip(' '))
        return sentence_list

    # 创建一个词典，包含数据集中所有单词。
    def create_vocab_list(self):
        sentence_list = self.clear_data()
        vocab_set = set([])
        for sentence in sentence_list:
            vocab_set = vocab_set | set(re.split(r'\W+', str.lower(sentence)))
        vocab_list = list(vocab_set)
        vocab_list.remove('')
        vocab_list.sort()
        vocab_file = np.array(vocab_list)
        np.save(self.vocab_save_path, vocab_file)
        print("vocab list created!")
        print("vocab list size:", vocab_file.size)

    # 加载词典
    def get_vocab_list(self):
        return np.load(self.vocab_save_path)

    # 每输入一个训练数据（句子），按照window_size创建词对，然后对联合矩阵进行更新
    def calculate_num(self, sentence, co_matrix, vocab_list):
        word_list = re.split(r'\W+', sentence)
        # if '' in word_list:
        #     word_list.remove('')
        word_list = [word for word in word_list if word != '']
        sentence_len = len(word_list)
        if sentence_len < 3: return
        for i in range(sentence_len):
            for j in range(self.window_size):
                if i+j+1 < sentence_len:
                    centre_word = word_list[i]
                    context_word = word_list[i+j+1]
                    centre_index = vocab_list.index(centre_word)
                    context_index = vocab_list.index(context_word)
                    co_matrix[centre_index, context_index] += 1
                if i-j-1 >= 0:
                    centre_word = word_list[i]
                    context_word = word_list[i-j-1]
                    centre_index = vocab_list.index(centre_word)
                    context_index = vocab_list.index(context_word)
                    co_matrix[centre_index, context_index] += 1

    # 创建联合矩阵
    def create_co_matrix(self):
        vocab_list = self.get_vocab_list()
        vocab_list = vocab_list.tolist()
        sentence_list = self.clear_data()
        vocab_len = len(vocab_list)
        co_matrix = np.zeros([vocab_len, vocab_len])
        for sentence in sentence_list:
            self.calculate_num(sentence, co_matrix, vocab_list)
        np.save(self.mat_save_path, co_matrix)
        print("co_matrix created!")

if __name__ == '__main__':
    native_data_path = 'datas/testPaper.txt'
    vocab_save_path = 'datas/vocab_list.npy'
    mat_save_path = 'datas/co_mat.npy'
    window_size = 5
    matrix = create_co_occurrence(native_data_path,vocab_save_path,mat_save_path,window_size)
    matrix.create_vocab_list()
    matrix.create_co_matrix()
