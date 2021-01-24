import numpy as np
import csv
import time

np.random.seed(1234)


def randomize():
    np.random.seed(time.time())


RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001


def load_abalone_dataset():
    rows = []
    with open("abalone.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
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


def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])


def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print(
                f"Epoch {epoch+1} : loss={np.mean(losses):5.3f}, accuracy={np.mean(accs):5.3f}/{acc:5.3f}"
            )

    final_acc = run_test(test_x, test_y)
    print(f"\nFinal Test : final accuracy = {final_acc:5.3f}")


def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count


def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * nth : mb_size * (nth + 1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]


def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy


def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x


def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b


def forward_postproc(output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_postproc(G_loss, diff):
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y) / y))
    return 1 - mdiff


def backprop_postproc_oneline(G_loss, diff):
    return 2 * diff / np.prod(diff.shape)


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


abalone_exec()

"""
Epoch 1 : loss=34.620, accuracy=0.540/0.791
Epoch 2 : loss=8.632, accuracy=0.807/0.805
Epoch 3 : loss=7.859, accuracy=0.808/0.805
Epoch 4 : loss=7.678, accuracy=0.808/0.809
Epoch 5 : loss=7.537, accuracy=0.811/0.809
Epoch 6 : loss=7.421, accuracy=0.811/0.811
Epoch 7 : loss=7.324, accuracy=0.811/0.813
Epoch 8 : loss=7.244, accuracy=0.812/0.814
Epoch 9 : loss=7.180, accuracy=0.813/0.813
Epoch 10 : loss=7.124, accuracy=0.812/0.813

Final Test : final accuracy = 0.813
"""
