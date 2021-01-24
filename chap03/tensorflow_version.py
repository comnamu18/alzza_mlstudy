# Code From : https://www.tensorflow.org/tutorials/load_data/csv
import tensorflow as tf

import csv
import numpy as np
import argparse

# Make numpy values easier to read.
SEED = 1234
np.random.seed(SEED)

RND_MEAN = 0
RND_STD = 0.0030


class CustomSoftmaxCrossEntropy(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        max_elm = tf.reduce_max(y_pred)
        diff = tf.transpose(tf.transpose(y_pred) - max_elm)
        exp = tf.math.exp(diff)
        sum_exp = tf.math.reduce_sum(exp)
        probs = tf.transpose(tf.transpose(exp) / sum_exp)
        return -tf.math.reduce_sum(y_true * tf.math.log(probs + 1.0e-10))


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses = logs["loss"]
        accs = logs["acc"]
        acc = logs["val_acc"]
        print(f"Epoch {epoch+1} : loss={losses:5.3f}, accuracy={accs:5.3f}/{acc:5.3f}")


def load_steel_dataset():
    rows = []
    with open("faults.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            rows.append(row)

    input_cnt, output_cnt = 27, 7
    data = np.asarray(rows, dtype="float32")
    return data, input_cnt, output_cnt


def init_model(input_cnt, output_cnt):
    initializer = tf.keras.initializers.RandomNormal(
        mean=RND_MEAN, stddev=RND_STD, seed=SEED
    )
    steel_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                output_cnt,
                input_shape=(input_cnt,),
                activation="softmax",
                bias_initializer="zeros",
                kernel_initializer=initializer,
            )
        ]
    )

    return steel_model


def train_and_test(steel_model, data, epoch_count, mb_size, output_cnt, learning_rate):
    shuffle_map, test_begin_idx = arrange_data(data, mb_size)
    test_dataset = get_test_data(data, shuffle_map, test_begin_idx, output_cnt)
    train_dataset = get_train_data(
        data, shuffle_map, test_begin_idx, output_cnt, mb_size
    )

    # CustomSoftmaxCrossEntropy()
    steel_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
        metrics=["acc"],
    )

    steel_model.fit(
        train_dataset,
        epochs=epoch_count,
        validation_data=test_dataset,
        callbacks=[CustomCallback()],
        verbose=0,
    )

    final_acc = steel_model.evaluate(test_dataset, verbose=0)
    print(f"\nFinal Test : final accuracy = {final_acc[1]:5.3f}")


def arrange_data(data, mb_size):
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size

    return shuffle_map, test_begin_idx


def get_test_data(data, shuffle_map, test_begin_idx, output_cnt):
    test_data = data[shuffle_map[test_begin_idx:]]
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data[:, :-output_cnt], test_data[:, -output_cnt:])
    )
    return test_dataset.batch(len(test_data))


def get_train_data(data, shuffle_map, test_begin_idx, output_cnt, mb_size):
    shuffle_size = len(shuffle_map) - test_begin_idx
    train_data = data[shuffle_map[:test_begin_idx]]
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_data[:, :-output_cnt], train_data[:, -output_cnt:])
    )
    return train_dataset.shuffle(shuffle_size).batch(mb_size)


def steel_exec(learning_rate, epoch_count=10, mb_size=10, report=1):
    data, input_cnt, output_cnt = load_steel_dataset()
    steel_model = init_model(input_cnt, output_cnt)
    train_and_test(steel_model, data, epoch_count, mb_size, output_cnt, learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Flag for adjust_ratio"
    )
    args = parser.parse_args()

    steel_exec(learning_rate=args.learning_rate)

"""
Epoch 1 : loss=2410812416.000, accuracy=0.286/0.440
Epoch 2 : loss=2418867712.000, accuracy=0.286/0.471
Epoch 3 : loss=2230692608.000, accuracy=0.315/0.448
Epoch 4 : loss=2257107456.000, accuracy=0.308/0.442
Epoch 5 : loss=2022994816.000, accuracy=0.320/0.361
Epoch 6 : loss=2154523136.000, accuracy=0.312/0.440
Epoch 7 : loss=2278467840.000, accuracy=0.330/0.176
Epoch 8 : loss=2407740160.000, accuracy=0.314/0.215
Epoch 9 : loss=2232590080.000, accuracy=0.329/0.453
Epoch 10 : loss=2301367296.000, accuracy=0.325/0.473

Final Test : final accuracy = 0.473

Epoch 1 : loss=221427584.000, accuracy=0.312/0.448
Epoch 2 : loss=213525968.000, accuracy=0.310/0.215
Epoch 3 : loss=244018672.000, accuracy=0.314/0.304
Epoch 4 : loss=232321984.000, accuracy=0.321/0.212
Epoch 5 : loss=222843136.000, accuracy=0.322/0.189
Epoch 6 : loss=219874016.000, accuracy=0.315/0.212
Epoch 7 : loss=233688720.000, accuracy=0.304/0.210
Epoch 8 : loss=234974640.000, accuracy=0.317/0.238
Epoch 9 : loss=225210848.000, accuracy=0.324/0.338
Epoch 10 : loss=235217696.000, accuracy=0.335/0.437

Final Test : final accuracy = 0.437
"""
