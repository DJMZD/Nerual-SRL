import sys

import numpy as np
import theano

from vocab import Vocab, UNK, PAD


def say(s, stream=sys.stdout):
    stream.write(s + "\n")
    stream.flush()


def say_flush(s, stream=sys.stdout):
    stream.write("\r" + s)
    stream.flush()


def load_conll_data(file_path):
    """Load data from conll2005

    :param
        file_path: The path to train/dev/test-set
    :return:
        A list
            1D: sent_num
            2D: sent_len
            element: (word, pos, syn, ne, pred, tag)
        For example:
        [
            [
                ("The",     "DT", "(S1(S(NP(NP*", "*", "-", "(A1*"),
                ("economy", "NN", "*",            "*", "-", "*"),
                ...
                (".",       ".",  "*))",          "*", "-", "*")
            ],
            ...
            [
                ("The",     "DT", "(S1(S(NP*",    "*", "-", ["(A1*", "(A1*")],
                ("trade",   "NN", "*",            "*", "-", ["*",    "*"]),
                ...
                ".",        ".",  "*))",          "*", "-", ["*",    "*"])
            ]
        ]
    """
    conll_data = []
    with open(file_path) as f:
        sent = []
        for line in f:
            line = line.strip().split()
            if len(line) == 5: continue
            if len(line) > 1:
                word = line[0]
                pos = line[1]
                syn = line[2]
                ne = line[3]
                pred = line[4]
                tag = line[5:]
                sent.append((word, pos, syn, ne, pred, tag))
            else:
                if sent: conll_data.append(sent)
                sent = []

    return conll_data


def load_word_dict_and_emb(emb_path):
    """Load word embeddings from the file

    :param
        emb_path: The path to word embeddings file
    :return:
        word_dict: Class Vocab, the vocabulary
        word_emb: A list of word embeddings(vector)
    """
    word_dict = Vocab()
    word_emb = []

    word_dict.convert(UNK)
    word_emb.append([])
    word_dict.convert(PAD)
    word_emb.append([])

    with open(emb_path) as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            word_dict.convert(word)
            if word == UNK:
                word_emb[0] = np.asarray(line[1:], dtype=theano.config.floatX)
            elif word == PAD:
                word_emb[1] = np.asarray(line[1:], dtype=theano.config.floatX)
            else:
                word_emb.append(np.asarray(line[1:], dtype=theano.config.floatX))

    return word_dict, np.asarray(word_emb, dtype=theano.config.floatX)


def get_samples_from_conll_data(conll_data,
                                word_dict,
                                tag_dict,
                                word_emb,
                                window_size):
    """Get samples from the return value from function load_conll_data

    :param
        conll_data: Return value from function load_conll_data
            1D: sent_num
            2D: sent_len
            element: (word, pos, syn, ne, pred, tag)
    :return:
        Samples for each predicate
            1D: sample_num
            2D: (features, tags)
            3D: sent_len
        For example: (word_id = [23, 1502, 2, 17] ,window_size = 3)
        [
            (
                (
                    # context, argument, predicate, mark
                    [(word emb for word_id2), (word emb for word_id17), ...],
                    [(word emb for word_id2), (word emb for word_id17), ...],
                    [(word emb for word_id2), (word emb for word_id17), ...],
                    [(word emb for word_id2), (word emb for word_id17), ...],
                ),
                (
                    # tags
                    [0, 0, 2, 4]
                )
            )
            ...
        ]
    """
    sys.stdout.write("\rReading data...\t\t0 / 90750")
    slide_len = window_size / 2

    samples = []
    for sent in conll_data:
        word_ids = [word_dict.convert(word[0].lower()) for word in sent]
        word_vectors = []
        for word_id in word_ids:
            if word_id:
                word_vectors.append(word_emb[word_id])
            else:
                word_vectors.append(word_emb[word_dict.convert(UNK)])
        predicate_indices = [i for (i, word) in enumerate(sent)
                             if word[4] != "-"]
        for i, predicate_index in enumerate(predicate_indices):
            # features
            # argument = word_vectors
            # predicate
            predicate = word_vectors[predicate_index]
            # context
            pads = [word_emb[word_dict.convert(PAD)] for _ in xrange(slide_len)]
            sent_with_pads = pads + word_vectors + pads
            context = []
            for j in xrange(window_size):
                context.extend(sent_with_pads[predicate_index + j])

            # mark
            marks = []
            for j in xrange(len(sent)):
                if (predicate_index - slide_len <= j <=
                    predicate_index + slide_len):
                    marks.append(1.0)
                else:
                    marks.append(0.0)
            # tag
            tags = []
            prev_tag = None
            for word in sent:
                tag = word[5][i]
                if tag.startswith("("):
                    if tag.endswith(")"):
                        bio_tag = "B-" + tag[1:-2]
                    else:
                        bio_tag = "B-" + tag[1:-1]
                        prev_tag = tag[1:-1]
                else:
                    if prev_tag:
                        bio_tag = "I-" + prev_tag
                        if tag.endswith(")"):
                            prev_tag = None
                    else:
                        bio_tag = "O"

                tags.append(bio_tag)

            sample_x = []
            for j in xrange(len(word_ids)):
                word_x = []
                word_x.extend(word_vectors[j])
                word_x.extend(predicate)
                word_x.extend(context)
                word_x.append(marks[j])
                sample_x.append(word_x)
            sample_y = [tag_dict.convert(tag) for tag in tags]

            samples.append((sample_x, sample_y))
            if len(samples) % 1000 == 0:
                sys.stdout.write(
                    "\rReading data...\t\t{} / 90750".format(len(samples))
                )
                sys.stdout.flush()

    sys.stdout.write(
        "\rReading data...\t\t{} / {}".format(len(samples), len(samples))
    )
    sys.stdout.write("\n")
    return samples


def get_batches_from_samples(samples,
                             word_dict,
                             tag_dict,
                             word_emb,
                             batch_size):
    """

    :param
        samples: Return value from function get_samples_from_conll_data
            1D: sample_num
            2D: (features, tags)
            3D: sent_len
        word_emb: initial word embeddings
    :return:
        A list of batches of samples
            1D: batch_num = ceil(sample_num * 1.0 / batch_size)
            2D: (batch_x, batch_y)
            3D: batch_size
            4D: longest sent_len in the batch
            element: word embedding or tag
    """
    batch_num = int(np.ceil(len(samples) * 1.0 / batch_size))
    sys.stdout.write("\rMaking batches...\t0 / {}".format(batch_num))

    np.random.shuffle(samples)
    samples.sort(key=lambda sample: len(sample[1]), reverse=True)

    batches = []
    batch_x = []
    batch_y = []
    longest_word_len = len(samples[0][0])

    for sample_x, sample_y in samples:
        if len(batch_y) == batch_size:
            batches.append((batch_x, batch_y))
            if len(batches) % 100 == 0:
                sys.stdout.write("\rMaking batches...\t{} / {}".format(
                    len(batches), batch_num)
                )
                sys.stdout.flush()
            batch_x = []
            batch_y = []
            longest_word_len = len(sample_y)
        for _ in xrange(longest_word_len - len(sample_x)):
            pad_x = []
            pad_x.extend(word_emb[word_dict.convert(PAD)])
            pad_x.extend(sample_x[0][len(word_emb[word_dict.convert(PAD)]):])
            sample_x.append(pad_x)
        batch_x.append(sample_x)
        for _ in xrange(longest_word_len - len(sample_y)):
            sample_y.append(tag_dict.convert("</s>"))
        batch_y.append(sample_y)

    if batch_y:
        batches.append((batch_x, batch_y))

    sys.stdout.write(
        "\rMaking batches...\t{} / {}".format(len(batches), len(batches))
    )
    sys.stdout.write("\n")
    return batches
