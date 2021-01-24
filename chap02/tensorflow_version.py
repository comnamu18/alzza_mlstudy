import tensorflow as tf
import argparse
import csv
import numpy as np

# Make numpy values easier to read.
SEED = 1234
np.random.seed(SEED)

RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001


class CustomCallback(tf.keras.callbacks.Callback):
    def safe_div(self, p, q):
        if np.abs(float(q)) < 1.0e-20:
            return np.sign(p)
        return float(p) / float(q)

    def on_epoch_end(self, epoch, logs=None):
        losses = logs["loss"]
        tp = logs["tp"]
        fp = logs["fp"]
        fn = logs["fn"]
        tn = logs["tn"]

        accuracy = self.safe_div(tp + tn, tp + tn + fp + fn)
        precision = self.safe_div(tp, tp + fp)
        recall = self.safe_div(tp, tp + fn)
        f1 = 2 * self.safe_div(recall * precision, recall + precision)

        acc_str = f"{accuracy:5.3f},{precision:5.3f},{recall:5.3f},{f1:5.3f}"
        print(f"Epoch {epoch+1} : loss={losses:5.3f}, result={acc_str}")
        self.losses = []


class CustomEvaluateCallback(tf.keras.callbacks.Callback):
    def safe_div(self, p, q):
        if np.abs(float(q)) < 1.0e-20:
            return np.sign(p)
        return float(p) / float(q)

    def on_test_end(self, logs=None):
        tp = logs["tp"]
        fp = logs["fp"]
        fn = logs["fn"]
        tn = logs["tn"]

        accuracy = self.safe_div(tp + tn, tp + tn + fp + fn)
        precision = self.safe_div(tp, tp + fp)
        recall = self.safe_div(tp, tp + fn)
        f1 = 2 * self.safe_div(recall * precision, recall + precision)

        acc_str = f"{accuracy:5.3f},{precision:5.3f},{recall:5.3f},{f1:5.3f}"

        print(f"Final Test : final accuracy={acc_str}")


class CustomSigMoidCrossEntropy(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return (
            tf.keras.activations.relu(y_pred)
            - y_pred * y_true
            + tf.math.log(1 + tf.math.exp(-tf.math.abs(y_pred)))
        )


def load_pulsar_dataset(adjust_ratio):
    pulsars, stars = [], []
    with open("pulsar_stars.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            if row[8] == "1":
                pulsars.append(row)
            else:
                stars.append(row)

    input_cnt, output_cnt = 8, 1

    star_cnt, pulsar_cnt = len(stars), len(pulsars)
    print(f"stars : {star_cnt}, pulsars : {pulsar_cnt}")
    if adjust_ratio:
        data = np.zeros([2 * star_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype="float32")
        for n in range(star_cnt):
            data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype="float32")
    else:
        data = np.zeros([pulsar_cnt + star_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype="float32")
        data[star_cnt:, :] = np.asarray(pulsars, dtype="float32")

    return data, input_cnt, output_cnt


def init_model(input_cnt, output_cnt):
    initializer = tf.keras.initializers.RandomNormal(
        mean=RND_MEAN, stddev=RND_STD, seed=SEED
    )
    pulsar_model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                output_cnt,
                activation="sigmoid",
                input_shape=(input_cnt,),
                bias_initializer="zeros",
                kernel_initializer=initializer,
            )
        ]
    )

    return pulsar_model


def train_and_test(pulsar_model, data, epoch_count, mb_size, output_cnt):
    shuffle_map, test_begin_idx = arrange_data(data, mb_size)
    test_dataset = get_test_data(data, shuffle_map, test_begin_idx, output_cnt)
    train_dataset = get_train_data(
        data, shuffle_map, test_begin_idx, output_cnt, mb_size
    )

    pulsar_model.compile(
        loss=CustomSigMoidCrossEntropy(),
        optimizer=tf.optimizers.SGD(),
        metrics=[
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.FalsePositives(name="fp"),
        ],
    )

    pulsar_model.fit(
        train_dataset,
        epochs=epoch_count,
        validation_data=test_dataset,
        callbacks=[CustomCallback()],
        verbose=0,
    )

    pulsar_model.evaluate(test_dataset, callbacks=[CustomEvaluateCallback()], verbose=0)


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


def pulsar_exec(adjust_ratio=False, epoch_count=10, mb_size=10, report=1):
    data, input_cnt, output_cnt = load_pulsar_dataset(adjust_ratio)
    abalone_model = init_model(input_cnt, output_cnt)
    train_and_test(abalone_model, data, epoch_count, mb_size, output_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjust_ratio", action="store_true", help="Flag for adjust_ratio"
    )
    args = parser.parse_args()

    pulsar_exec(adjust_ratio=args.adjust_ratio)

"""
--adjust_ratio
Epoch 1 : loss=0.546, result=0.901,0.953,0.844,0.895
Epoch 2 : loss=0.541, result=0.911,0.962,0.857,0.906
Epoch 3 : loss=0.540, result=0.914,0.962,0.862,0.909
Epoch 4 : loss=0.540, result=0.915,0.966,0.860,0.910
Epoch 5 : loss=0.540, result=0.914,0.963,0.862,0.909
Epoch 6 : loss=0.539, result=0.916,0.965,0.863,0.911
Epoch 7 : loss=0.538, result=0.917,0.966,0.864,0.912
Epoch 8 : loss=0.538, result=0.917,0.967,0.864,0.913
Epoch 9 : loss=0.538, result=0.917,0.966,0.865,0.913
Epoch 10 : loss=0.538, result=0.917,0.967,0.863,0.912
Final Test : final accuracy=0.914,0.977,0.850,0.909
"""
