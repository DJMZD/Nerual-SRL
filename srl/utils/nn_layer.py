import numpy as np
import theano
import theano.tensor as T

from nn_utils import build_shared_zeros, get_uniform_weight, logsumexp


class LSTMUnit(object):
    """LSTM unit for nn"""
    def __init__(self, input_dim, hidden_dim, activation=T.tanh):
        self.c_0 = build_shared_zeros(hidden_dim)
        self.h_0 = activation(self.c_0)
        self.activation = activation
        self.W = theano.shared(get_uniform_weight(input_dim, hidden_dim))
        self.W_i = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.U_i = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.V_i = theano.shared(get_uniform_weight(hidden_dim))
        self.W_f = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.U_f = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.V_f = theano.shared(get_uniform_weight(hidden_dim))
        self.W_c = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.U_c = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.W_o = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.U_o = theano.shared(get_uniform_weight(hidden_dim, hidden_dim))
        self.V_o = theano.shared(get_uniform_weight(hidden_dim))

        self.parameters = [self.W,
                           self.W_f, self.U_f, self.V_f,
                           self.W_i, self.U_i, self.V_i,
                           self.W_c, self.U_c,
                           self.W_o, self.U_o, self.V_o]

    def forward(self, x_t, h_tm1, c_tm1):
        """Calculate LSTM unit

        :param
            x_t: input
                1D: batch_size
                2D: input_dim
            h_tm1: previous output
                1D: batch_size
                2D: hidden_dim
            c_tm1: previous memory cell
                1D: batch_size
                2D: hidden_dim
        :return
            h_t: output
                1D: batch_size
                2D: hidden_dim
            c_t: memory cell
                1D: batch_size
                2D: hidden_dim
        """
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_i)
                             + T.dot(h_tm1, self.U_i)
                             + c_tm1 * self.V_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_f)
                             + T.dot(h_tm1, self.U_f)
                             + c_tm1 * self.V_f)
        c_hat_t = self.activation(T.dot(x_t, self.W_c)
                                  + T.dot(h_tm1, self.U_c))
        c_t = f_t * c_tm1 + i_t * c_hat_t
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_o)
                             + T.dot(h_tm1, self.U_o)
                             + c_t * self.V_o)
        h_t = o_t * self.activation(c_t)

        return h_t, c_t


class GRUUnit(object):
    pass


class CRF(object):
    def __init__(self, input_dim, hidden_dim):
        self.W = theano.shared(get_uniform_weight(input_dim, hidden_dim))
        self.W_transition = theano.shared(get_uniform_weight(hidden_dim,
                                                             hidden_dim))
        self.BOS_probability = theano.shared(get_uniform_weight(hidden_dim))

        self.parameters = [self.W, self.W_transition, self.BOS_probability]

    def forward_probability(self, h_t, y_t,
                            y_tm1, y_score_tm1, z_score_tm1):
        """Calculate CRF unit

        :param
            h_t: emission
                1D: batch_size
                2D: output_dim
            y_t: tag
                1D: batch_size
            y_tm1: previous tag
            y_score_tm1: log likelihood of previous tag
                1D: batch_size
            z_score_tm1: sum of all log likelihood of all previous tags
                1D: batch_size
                2D: output_dim
        :return
            y_t: tag
                1D: batch_size
            y_score_t: log likelihood of tag
                1D: batch_size
            z_score_t: sum of all log likelihood of all tags
                1D: batch_size
                2D: output_dim
        """
        batch_size = T.cast(h_t.shape[0], dtype="int32")
        y_score_t = (y_score_tm1                                  # forward
                     + self.W_transition[y_t, y_tm1]              # transition
                     + h_t[T.arange(batch_size), y_t])            # emission
        z_score_t = (logsumexp(z_score_tm1.dimshuffle(0, 'x', 1)  # forward
                               + self.W_transition,               # transition
                               axis=2).reshape(h_t.shape)
                     + h_t)                                       # emission

        return y_t, y_score_t, z_score_t

    def get_log_probabilities(self, h, y):
        """Calculate log probabilities of y(predicated/gold tags)

        :param
            h: outputs from previous layer
                1D: sent_len
                2D: batch_size
                3D: output_dim
            y: predicated tags
                1D: sent_len
                2D: batch_size
            batch_size: batch size
        :return: log probabilities of y
        """
        batch_size = T.cast(y.shape[1], dtype="int32")
        # log likelihood of 1st tags
        # 1D: batch_size
        y_score_0 = (self.BOS_probability[y[0]]
                     + h[0][T.arange(batch_size), y[0]])
        # sum of all log likelihood of 1st all tags
        # 1D: batch_size, 2D: output_dim
        z_score_0 = self.BOS_probability + h[0]
        [_, y_score, z_score], _ = theano.scan(
            fn=self.forward_probability,
            sequences=[h[1:], y[1:]],
            outputs_info=[y[0], y_score_0, z_score_0],
        )

        return y_score[-1] - logsumexp(z_score[-1], axis=1).flatten()

    def forward_viterbi(self, h_t, score_tm1):
        """Calculate viterbi score(best log likelihood)

        :param
            h_t: emission
                1D: batch_size
                2D: output_dim
            score_tm1: previous viterbi tag
                1D: batch_size
                2D: output_dim
        :return
            score_t: viterbi score of tag
                1D: batch_size
                2D: output_dim
            best_tag: previous viterbi tag
                1D: batch_size
                2D: output_dim
        """
        # scores[i][j]: log likelihood of tag i with previous tag j
        # 1D: batch_size
        # 2D: output_dim
        # 3D: output_dim
        scores = (score_tm1.dimshuffle(0, 'x', 1)
                  + self.W_transition
                  + h_t.dimshuffle(0, 1, 'x'))
        score_t, best_tag = T.max_and_argmax(scores, axis=2)

        return score_t, T.cast(best_tag, dtype="int32")

    def backward_viterbi(self, tag_t, best_tag_tm1):
        """Get viterbi tag(with best score)

        :param
            tag_t: emission
                1D: batch_size
                2D: output_dim
            best_tag_tm1: previous viterbi tag
                1D: batch_size
        :return: viterbi tag
                1D: batch_size
        """
        batch_size = T.cast(tag_t.shape[0], dtype="int32")

        return tag_t[T.arange(batch_size), best_tag_tm1]

    def get_viterbi_tags(self, h):
        """Get viterbi tags

        :param
            h: outputs from previous layer
                1D: sent_len
                2D: batch_size
                3D: output_dim
        :return: #viterbi scores of h
        """
        score_0 = self.BOS_probability + h[0]
        # 1D: sent_len
        # 2D: batch_size
        # 3D: output_dim
        [best_scores, best_tags], _ = theano.scan(
            fn=self.forward_viterbi,
            sequences=[h[1:]],
            outputs_info=[score_0, None]
        )
        # 1D: batch_size
        best_last_tags = T.cast(T.argmax(best_scores[-1], axis=1),
                                  dtype="int32")
        # 1D: sent_len
        # 2D: batch_size
        best_tags, _ = theano.scan(fn=self.backward_viterbi,
                                     sequences=best_tags[::-1],
                                     outputs_info=best_last_tags)
        best_tags = T.concatenate([best_tags[::-1].dimshuffle(1, 0),
                                    best_last_tags.dimshuffle(0, "x")],
                                    axis=1)

        return best_tags
