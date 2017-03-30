import numpy as np
import theano
import argparse
import train

theano.config.floatX = "float32"
# theano.config.optimizer = "None"
# theano.config.exception_verbosity="high"


#if __name__ == "main":


parser = argparse.ArgumentParser(description="SRL tagger.")
parser.add_argument("--train_data_path", help="path to training data")
parser.add_argument("--dev_data_path", help="path to dev data")
parser.add_argument("--test_data_path", help="path to test data")
parser.add_argument("--init_emb_path", help="path to init word embeddings")
parser.add_argument("--unit", type=str, default="lstm", help="unit type of nn")
parser.add_argument("--depth", type=int, default=1, help="number of layers")
parser.add_argument("--epoch", type=int, default=50, help="number of layers")
parser.add_argument("--regulation", type=float, default=0.0001, help="regulation rate")
parser.add_argument("--hidden_dim", type=int, default=32, help="dimension of hidden layers")
parser.add_argument("--window_size", type=int, default=5, help="window size")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")

argv = parser.parse_args()

train.main(argv)
