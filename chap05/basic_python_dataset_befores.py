from basic_python_dataset import Dataset
from basic_python_mathutil import *  # noqa


class AbaloneDataset(Dataset):
    def __init__(self):
        super(AbaloneDataset, self).__init__("abalone", "regression")

        rows, _ = load_csv("../chap01/abalone.csv")

        xs = np.zeros([len(rows), 10])
        ys = np.zeros([len(rows), 1])

        for n, row in enumerate(rows):
            if row[0] == "I":
                xs[n, 0] = 1
            if row[0] == "M":
                xs[n, 1] = 1
            if row[0] == "F":
                xs[n, 2] = 1
            xs[n, 3:] = row[1:-1]
            ys[n, :] = row[-1:]

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, "%4.2f")
            print("{} => 추정 {:4.1f} : 정답 {:4.1f}".format(xstr, est[0], ans[0]))


class PulsarDataset(Dataset):
    def __init__(self):
        super(PulsarDataset, self).__init__("pulsar", "binary")

        rows, _ = load_csv("../chap02/pulsar_stars.csv")

        data = np.asarray(rows, dtype="float32")
        self.shuffle_data(data[:, :-1], data[:, -1:], 0.8)
        self.target_names = ["별", "펄서"]

    def visualize(self, xs, estimates, answers):
        for n in range(len(xs)):
            x, est, ans = xs[n], estimates[n], answers[n]
            xstr = vector_to_str(x, "%5.1f", 3)
            estr = self.target_names[int(round(est[0]))]
            astr = self.target_names[int(round(ans[0]))]
            rstr = "O"
            if estr != astr:
                rstr = "X"
            print(
                "{} => 추정 {}(확률 {:4.2f}) : 정답 {} => {}".format(
                    xstr, estr, est[0], astr, rstr
                )
            )


class SteelDataset(Dataset):
    def __init__(self):
        super(SteelDataset, self).__init__("steel", "select")

        rows, headers = load_csv("../chap03/faults.csv")

        data = np.asarray(rows, dtype="float32")
        self.shuffle_data(data[:, :-7], data[:, -7:], 0.8)

        self.target_names = headers[-7:]

    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)


class PulsarSelectDataset(Dataset):
    def __init__(self):
        super(PulsarSelectDataset, self).__init__("pulsarselect", "select")

        rows, _ = load_csv("../chap02/pulsar_stars.csv")

        data = np.asarray(rows, dtype="float32")
        self.shuffle_data(data[:, :-1], onehot(data[:, -1], 2), 0.8)
        self.target_names = ["별", "펄서"]

    def visualize(self, xs, estimates, answers):
        show_select_results(estimates, answers, self.target_names)
