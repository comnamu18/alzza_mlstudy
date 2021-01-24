import numpy as np
import csv
import argparse

np.random.seed(1234)
RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001


def init_model_hidden1():
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt

    pm_hidden = alloc_param_pair([input_cnt, hidden_cnt])
    pm_output = alloc_param_pair([hidden_cnt, output_cnt])


def alloc_param_pair(shape):
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])
    return {"w": weight, "b": bias}


def forward_neuralnet_hidden1(x):
    global pm_output, pm_hidden

    hidden = relu(np.matmul(x, pm_hidden["w"]) + pm_hidden["b"])
    output = np.matmul(hidden, pm_output["w"]) + pm_output["b"]

    return output, [x, hidden]


def relu(x):
    return np.maximum(x, 0)


def backprop_neuralnet_hidden1(G_output, aux):
    global pm_output, pm_hidden

    x, hidden = aux

    g_output_w_out = hidden.transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    g_output_hidden = pm_output["w"].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output["w"] -= LEARNING_RATE * G_w_out
    pm_output["b"] -= LEARNING_RATE * G_b_out

    G_hidden = G_hidden * relu_derv(hidden)

    g_hidden_w_hid = x.transpose()
    G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
    G_b_hid = np.sum(G_hidden, axis=0)

    pm_hidden["w"] -= LEARNING_RATE * G_w_hid
    pm_hidden["b"] -= LEARNING_RATE * G_b_hid


def relu_derv(y):
    return np.sign(y)


def init_model_hiddens():
    global pm_output, pm_hiddens, input_cnt, output_cnt, hidden_config

    pm_hiddens = []
    prev_cnt = input_cnt

    for hidden_cnt in hidden_config:
        pm_hiddens.append(alloc_param_pair([prev_cnt, hidden_cnt]))
        prev_cnt = hidden_cnt

    pm_output = alloc_param_pair([prev_cnt, output_cnt])


def forward_neuralnet_hiddens(x):
    global pm_output, pm_hiddens

    hidden = x
    hiddens = [x]

    for pm_hidden in pm_hiddens:
        hidden = relu(np.matmul(hidden, pm_hidden["w"]) + pm_hidden["b"])
        hiddens.append(hidden)

    output = np.matmul(hidden, pm_output["w"]) + pm_output["b"]

    return output, hiddens


def backprop_neuralnet_hiddens(G_output, aux):
    global pm_output, pm_hiddens

    hiddens = aux

    g_output_w_out = hiddens[-1].transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)

    g_output_hidden = pm_output["w"].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)

    pm_output["w"] -= LEARNING_RATE * G_w_out
    pm_output["b"] -= LEARNING_RATE * G_b_out

    for n in reversed(range(len(pm_hiddens))):
        G_hidden = G_hidden * relu_derv(hiddens[n + 1])

        g_hidden_w_hid = hiddens[n].transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)

        g_hidden_hidden = pm_hiddens[n]["w"].transpose()
        G_hidden = np.matmul(G_hidden, g_hidden_hidden)

        pm_hiddens[n]["w"] -= LEARNING_RATE * G_w_hid
        pm_hiddens[n]["b"] -= LEARNING_RATE * G_b_hid


global hidden_config


def init_model():
    if hidden_config is not None:
        print("은닉 계층 {}개를 갖는 다층 퍼셉트론이 작동되었습니다.".format(len(hidden_config)))
        init_model_hiddens()
    else:
        print("은닉 계층 하나를 갖는 다층 퍼셉트론이 작동되었습니다.")
        init_model_hidden1()


def forward_neuralnet(x):
    if hidden_config is not None:
        return forward_neuralnet_hiddens(x)
    else:
        return forward_neuralnet_hidden1(x)


def backprop_neuralnet(G_output, hiddens):
    if hidden_config is not None:
        backprop_neuralnet_hiddens(G_output, hiddens)
    else:
        backprop_neuralnet_hidden1(G_output, hiddens)


def set_hidden(info):
    global hidden_cnt, hidden_config
    if isinstance(info, int):
        hidden_cnt = info
        hidden_config = None
    else:
        hidden_config = info


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


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="abalone", help="Flag for adjust_ratio"
    )
    args = parser.parse_args()

    if args.mode == "abalone":
        set_hidden([])
        abalone_exec()

"""
python basic_python.py --adjust_ratio
Epoch 1 : loss=0.413, result=0.924,0.932,0.916,0.924
Epoch 2 : loss=0.380, result=0.920,0.976,0.862,0.915
Epoch 3 : loss=0.374, result=0.899,0.872,0.937,0.903
Epoch 4 : loss=0.382, result=0.870,0.825,0.942,0.879
Epoch 5 : loss=0.376, result=0.910,0.895,0.930,0.912
Epoch 6 : loss=0.367, result=0.828,0.757,0.969,0.850
Epoch 7 : loss=0.367, result=0.900,0.875,0.935,0.904
Epoch 8 : loss=0.367, result=0.890,0.994,0.786,0.878
Epoch 9 : loss=0.371, result=0.925,0.934,0.916,0.925
Epoch 10 : loss=0.367, result=0.598,0.556,0.997,0.714

Final Test : final accuracy = 0.598,0.556,0.997,0.714
"""
