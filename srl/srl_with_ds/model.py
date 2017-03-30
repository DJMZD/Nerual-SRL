import sys
import time

import numpy as np
import theano
import theano.tensor as T

from srl.utils.io_utils import say, say_flush
from srl.utils.nn_layer import LSTMUnit, GRUUnit, CRF
from srl.utils.nn_utils import norm
from srl.utils.optimizer import SGDOptimizer, AdamOptimizer
from srl.utils.evaluator import calculate_accuracy, calculate_f_value


class Model(object):
    def __init__(self, argv, word_dict, tag_dict, init_emb):
        say("Building model...")
        self.parameters = []
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.init_emb = init_emb
        # layers
        self.layers = []
        if argv.unit.lower() == "lstm":
            layer = LSTMUnit
        else:
            layer = GRUUnit
        for i in xrange(argv.depth):
            if i == 0:
                self.layers.append(layer(
                    (argv.window_size + 2) * init_emb.shape[1] + 1,
                    argv.hidden_dim
                ))
            else:
                self.layers.append(layer(
                    2 * argv.hidden_dim,
                    argv.hidden_dim
                ))
            self.parameters += self.layers[i].parameters
        # crf
        self.crf = CRF(2 * argv.hidden_dim, len(tag_dict.i2w))
        self.parameters += self.crf.parameters
        self.parameters_num = sum(len(p.get_value(borrow=True).ravel())
                                  for p in self.parameters)
        say("\tNum of parameters: {}".format(self.parameters_num))
        # model
        x = T.ftensor3("x")
        y = T.imatrix("y")
        h = self.forward(x, argv.hidden_dim)
        viterbi_tags = self.crf.get_viterbi_tags(h)
        error = T.neq(y, viterbi_tags)
        log_probability = self.crf.get_log_probabilities(h,
                                                         y.dimshuffle(1, 0))
        neg_log_likelihood = -T.mean(log_probability)
        cost = (neg_log_likelihood
                + argv.regulation * norm(self.parameters, 2) ** 2 / 2)
        optimizer = AdamOptimizer(self.parameters)
        updates = optimizer.update(cost)
        # functions
        self.train_f = theano.function(inputs=[x, y],
                                       outputs=[error, neg_log_likelihood],
                                       updates=updates,
                                       mode="FAST_RUN")
        self.predict_f = theano.function(inputs=[x, y],
                                         outputs=[viterbi_tags, error],
                                         mode="FAST_RUN")
        sys.stdout.write("\n")

    def forward(self, x, hidden_dim):
        """Forward for the whole network

        :param
            x: input
                1D: batch_size
                2D: sent_len
                3D: input_dim
            batch_size: batch size
            hidden_dim: hidden dim
        :return:
            1D: sent_len
            2D: batch_size
            3D: output_dim(tag size)
        """
        batch_size = T.cast(x.shape[0], dtype="int32")
        x = x.dimshuffle(1, 0, 2)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = T.maximum(0, T.dot(x, layer.W))
                h_0 = layer.h_0 * T.ones((batch_size, hidden_dim))
                c_0 = layer.c_0 * T.ones((batch_size, hidden_dim))
            else:
                x = T.maximum(0, T.dot(T.concatenate([x, h], 2), layer.W))
                x = x[::-1]
                h_0 = x[0]
                c_0 = c[-1]
            # x: 1D: sent_len, 2D: batch_size, 3D: (2 * )hidden_dim
            # h_0: 1D: batch_size, 2D: hidden_dim
            # c_0: 1D: batch_size, 2D: hidden_dim
            [h, c], _ = theano.scan(fn=layer.forward,
                                    sequences=[x],
                                    outputs_info=[h_0, c_0])
            # h: 1D: sent_len, 2D: batch_size, 3D: hidden_dim
            # c: 1D: sent_len, 2D: batch_size, 3D: hidden_dim
        h = T.maximum(0, T.dot(T.concatenate([x, h], 2), self.crf.W))
        if len(self.layers) % 2 == 0:
            h = h[::-1]

        return h

    def get_score(self, h):
        pass

    def train(self, train_batches):
        say("\tTrain:")
        batch_num = len(train_batches)
        say_flush("\t\tTraining data...\t0 / {}".format(batch_num))
        batch_indices = range(len(train_batches))
        np.random.shuffle(batch_indices)

        start_time = time.time()
        errors = []
        losses = []
        for i, batch_index in enumerate(batch_indices, 1):
            if i % 100 == 0:
                say_flush("\t\tTraining data...\t{} / {}".format(i, batch_num))
            batch_x, batch_y = train_batches[batch_index]
            error, loss = self.train_f(batch_x, batch_y)
            errors.extend(error)
            losses.append(loss)
        say_flush("\t\tTraining data...\t{} / {}".format(batch_num, batch_num))
        print
        end_time = time.time()
        accuracy = calculate_accuracy(errors)

        say("\t\tTime: {} secs".format(end_time - start_time))
        say("\t\tAvg nll: {}".format(np.mean(losses)))
        say("\t\tAccuracy: {}".format(accuracy))

    def predict(self, dev_samples, tag_dict, mode):
        print("\tmode:")
        sample_num = len(dev_samples)
        say_flush("\r\t\tPredicting data...\t0 / {}".format(sample_num))

        start_time = time.time()
        predict_tags = []
        gold_tags = []
        errors = []
        for i, (sample_x, sample_y) in enumerate(dev_samples, 1):
            if i % 100 == 0:
                say_flush(
                    "\t\tPredicting data...\t{} / {}".format(i, sample_num)
                )
            viterbi_tags, error = self.predict_f([sample_x], [sample_y])
            predict_tags.append(viterbi_tags[0])
            gold_tags.append(sample_y)
            errors.append(error[0])
        say_flush(
            "\t\tPredicting data...\t{} / {}".format(sample_num, sample_num)
        )
        say("\t\t")
        end_time = time.time()
        accuracy = calculate_accuracy(errors)
        correct_span_num, predict_tag_num, gold_span_num,\
            precision, recall, f_value = calculate_f_value(
                predict_tags, gold_tags, tag_dict
            )

        say("\t\tTime: {} secs".format(end_time - start_time))
        say("\t\tAccuracy: {}".format(accuracy))
        say("\t\t{} Correct, Predict, Gold: {}, {}, {}".format(
            mode, correct_span_num, predict_tag_num, gold_span_num
        ))
        say("\t\t{} Precision, Recall, F value: {}, {}, {}".format(
            mode, precision, recall, f_value
        ))

        return f_value
