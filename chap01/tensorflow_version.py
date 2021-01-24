# Code From : https://www.tensorflow.org/tutorials/load_data/csv
import tensorflow as tf

import csv
import numpy as np

# Make numpy values easier to read.
SEED = 1234
np.random.seed(SEED)

RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        losses = logs["loss"]
        accs = logs["CustomMetric"]
        acc = logs["val_CustomMetric"]
        print(f"Epoch {epoch+1} : loss={losses:5.3f}, accuracy={accs:5.3f}/{acc:5.3f}")


class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name="CustomMetric", **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.accuracy = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.accuracy.append(
            1 - tf.reduce_mean(tf.math.abs((y_pred - y_true) / y_true))
        )

    def result(self):
        return_value = tf.reduce_mean(self.accuracy)
        self.accuracy = []
        return return_value


def load_abalone_dataset():
    rows = []
    cols = []
    with open("abalone.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        cols = next(csvreader, None)
        for row in csvreader:
            rows.append(row)

    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + output_cnt])
    for n, row in enumerate(rows):
        if row[0] == "l":
            data[n, 0] = 1
        if row[0] == "M":
            data[n, 1] = 1
        if row[0] == "F":
            data[n, 2] = 1
        data[n, 3:] = row[1:]
    cols = ["l", "M", "F"] + cols[1:]

    return data, input_cnt, output_cnt


def init_model(input_cnt, output_cnt):
    initializer = tf.keras.initializers.RandomNormal(
        mean=RND_MEAN, stddev=RND_STD, seed=SEED
    )
    abalone_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                output_cnt,
                input_shape=(input_cnt,),
                bias_initializer="zeros",
                kernel_initializer=initializer,
            )
        ]
    )

    return abalone_model


def train_and_test(abalone_model, data, epoch_count, mb_size, output_cnt):
    shuffle_map, test_begin_idx = arrange_data(data, mb_size)
    test_dataset = get_test_data(data, shuffle_map, test_begin_idx, output_cnt)
    train_dataset = get_train_data(
        data, shuffle_map, test_begin_idx, output_cnt, mb_size
    )

    abalone_model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.SGD(),
        metrics=[CustomMetric()],
    )

    abalone_model.fit(
        train_dataset,
        epochs=epoch_count,
        validation_data=test_dataset,
        callbacks=[CustomCallback()],
        verbose=0,
    )

    final_acc = abalone_model.evaluate(test_dataset, verbose=0)
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


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    data, input_cnt, output_cnt = load_abalone_dataset()
    abalone_model = init_model(input_cnt, output_cnt)
    train_and_test(abalone_model, data, epoch_count, mb_size, output_cnt)


abalone_exec()

"""
poch 1 : loss=10.204, accuracy=0.803/0.810
Epoch 2 : loss=6.904, accuracy=0.819/0.814
Epoch 3 : loss=6.751, accuracy=0.840/0.811
Epoch 4 : loss=6.649, accuracy=0.857/0.811
Epoch 5 : loss=6.569, accuracy=0.871/0.810
Epoch 6 : loss=6.491, accuracy=0.838/0.822
Epoch 7 : loss=6.429, accuracy=0.793/0.809
Epoch 8 : loss=6.360, accuracy=0.818/0.817
Epoch 9 : loss=6.282, accuracy=0.843/0.810
Epoch 10 : loss=6.223, accuracy=0.862/0.822

Final Test : final accuracy = 0.822
"""
