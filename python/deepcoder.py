import numpy as np
import tensorflow as tf
import json, argparse, copy, os
import model
from util import preprocess_json, print_value, convert_example
params = tf.flags.FLAGS

tf.flags.DEFINE_integer("learning_rate", 0.01, "")
tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_integer("num_epoch", 100, "")
tf.flags.DEFINE_integer("num_input", 3, "")
tf.flags.DEFINE_integer("embedding_size", 20, "")
tf.flags.DEFINE_integer("integer_min", -100, "")
tf.flags.DEFINE_integer("integer_max", 100, "")
tf.flags.DEFINE_integer("num_example", 5, "")
tf.flags.DEFINE_integer("num_hidden_layer", 3, "")
tf.flags.DEFINE_integer("max_list_len", 10, "")
tf.flags.DEFINE_integer("hidden_layer_size", 256, "")
tf.flags.DEFINE_integer("attribute_size", 34, "")


parser = argparse.ArgumentParser(description='DeepCoder Model')
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('dir', type=str, help='Path to data')
parser.add_argument('-e', '--enable', choices=['true','false'],default='true',help='Whether to use model for searching')
args = parser.parse_args()

params_dict = {'num_input': params.num_input,
                'integer_min': params.integer_min,
                'integer_range': params.integer_max - params.integer_min + 1,
                'max_list_len': params.max_list_len}


def train():
    # Load dataset from json
    print("Loading dataset")
    file = open(args.dir, 'r')
    x = json.load(file)
    y = preprocess_json(x, params_dict)
    data = np.asarray([y[i][0] for i in range(len(y))])
    target = np.asarray([y[i][1] for i in range(len(y))])
    print("Complete")

    d = model.DeepCoder(num_input = params.num_input,
                        embedding_size = params.embedding_size,
                        integer_range = params.integer_max - params.integer_min + 1,
                        num_example = params.num_example,
                        max_list_len = params.max_list_len,
                        num_hidden_layer = params.num_hidden_layer,
                        hidden_layer_size = params.hidden_layer_size,
                        attribute_size = params.attribute_size,
                        batch_size = params.batch_size,
                        num_epoch = params.num_epoch,
                        learning_rate = params.learning_rate)
    d.train(data, target)

def predict():
    # Load example
    file = open(args.dir, 'r')
    data = json.load(file)
    data_backup = copy.deepcopy(data)
    data_processed = np.array([convert_example(x, params_dict) for x in data])

    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    try:
        os.remove(output_path)
    except OSError:
        pass
    output = open(output_path, 'w+')

    # To produce standard input of c++ parser, the output has to be the same
    for x in data_backup:
        for v in x["input"]:
            print_value(v, output)
        print("---", file=output)
        print_value(x["output"], output)
        print("---", file=output)
    print("---", file=output)

    x = 'Attribute: '
    if args.enable == 'true':
        d = model.DeepCoder(num_input = params.num_input,
            embedding_size = params.embedding_size,
            integer_range = params.integer_max - params.integer_min + 1,
            num_example = params.num_example,
            max_list_len = params.max_list_len,
            num_hidden_layer = params.num_hidden_layer,
            hidden_layer_size = params.hidden_layer_size,
            attribute_size = params.attribute_size,
            batch_size = params.batch_size,
            num_epoch = params.num_epoch,
            learning_rate = params.learning_rate)

        attributes = d.predict(np.asarray([data_processed],dtype=np.int32))
        for t in attributes.tolist()[0]:
            x += str(t) + " "
    else:
        x += "1 " * params.attribute_size
    print(x, file=output)
    output.close()


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        predict()

if __name__ == '__main__':
    main()
