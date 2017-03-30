import numpy as np
import theano

from model import Model
from srl.utils.io_utils import (say, say_flush,
                                load_conll_data, load_word_dict_and_emb,
                                get_samples_from_conll_data,
                                get_batches_from_samples)
from srl.utils.vocab import Vocab


def main(argv):
    tag_dict = Vocab()
    srl_base_tags = [
        "V",
        "A0", "A1", "A2", "A3", "A4", "A5", "AA",
        "AM", "AM-ADV", "AM-CAU", "AM-DIR", "AM-DIS",
        "AM-EXT", "AM-LOC", "AM-MNR", "AM-MOD", "AM-NEG",
        "AM-PNC","AM-PRD", "AM-REC", "AM-TMP", "AM-TM",
    ]
    tag_dict.convert("<s>")
    tag_dict.convert("</s>")
    tag_dict.convert("O")
    for tag in srl_base_tags:
        tag_dict.convert("B-" + tag)
        tag_dict.convert("I-" + tag)
        tag_dict.convert("B-C-" + tag)
        tag_dict.convert("I-C-" + tag)
        tag_dict.convert("B-R-" + tag)
        tag_dict.convert("I-R-" + tag)
    tag_dict.freeze()

    train_corpus = load_conll_data(argv.train_data_path)
    dev_corpus = load_conll_data(argv.dev_data_path)
    test_corpus = load_conll_data(argv.test_data_path)
    word_dict, word_emb = load_word_dict_and_emb(argv.init_emb_path)
    word_dict.freeze()
    train_samples = get_samples_from_conll_data(train_corpus,
                                                word_dict,
                                                tag_dict,
                                                word_emb,
                                                argv.window_size)
    dev_samples = get_samples_from_conll_data(dev_corpus,
                                              word_dict,
                                              tag_dict,
                                              word_emb,
                                              argv.window_size)
    test_samples = get_samples_from_conll_data(test_corpus,
                                               word_dict,
                                               tag_dict,
                                               word_emb,
                                               argv.window_size)
    train_batches = get_batches_from_samples(train_samples,
                                             word_dict,
                                             tag_dict,
                                             word_emb,
                                             argv.batch_size)
    model = Model(argv, word_dict, tag_dict, word_emb)

    best_dev_f_value = 0.
    best_test_f_value = 0.
    for epoch in range(argv.epoch):
        say("Epoch {}:".format(epoch + 1))
        # train
        model.train(train_batches)
        # dev
        dev_f_value = model.predict(dev_samples, tag_dict, "Dev")
        if dev_f_value > best_dev_f_value:
            best_dev_f_value = dev_f_value
        # test
        test_f_value = model.predict(test_samples, tag_dict, "Test")
        if dev_f_value == best_dev_f_value:
            best_test_f_value = test_f_value

    say("Best dev F value: {}".format(best_dev_f_value))
    say("Best test F value: {}".format(best_test_f_value))
