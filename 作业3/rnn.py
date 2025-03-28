import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        # 定义LSTM层
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,  # 输入特征维度
            hidden_size=lstm_hidden_dim,  # 隐藏层维度
            num_layers=2,  # 2层LSTM
            batch_first=True  # 输入输出格式为(batch, seq, feature)
        )

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)  # 调用权重初始化函数
        self.softmax = nn.LogSoftmax(dim=1)  # 修改为指定dim=1

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)

        # LSTM前向传播
        # 初始化隐藏状态和细胞状态为0
        h0 = torch.zeros(2, batch_input.size(0), self.lstm_dim)  # 2层LSTM
        c0 = torch.zeros(2, batch_input.size(0), self.lstm_dim)

        # 输入LSTM
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))

        out = output.contiguous().view(-1, self.lstm_dim)
        out = F.relu(self.fc(out))
        out = self.softmax(out)

        if is_test:
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            output = out

        return output