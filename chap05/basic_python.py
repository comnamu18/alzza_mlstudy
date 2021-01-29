from basic_python_model import *  # noqa
from basic_python_dataset_befores import *  # noqa
from basic_python_dataset_flower import *  # noqa
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="Flag for adjust_ratio")
    args = parser.parse_args()

    if args.mode == 0:
        ad = AbaloneDataset()
        am = MlpModel("abalone_model", ad, [])
        am.exec_all(epoch_count=10, report=2)
    elif args.mode == 1:
        pd = PulsarDataset()
        pm = MlpModel("pulsar_model", pd, [4])
        pm.exec_all()
        pm.visualize(5)
    elif args.mode == 2:
        sd = SteelDataset()
        sm = MlpModel("steel_model", sd, [12, 7])
        sm.exec_all(epoch_count=50, report=10)
    elif args.mode == 3:
        psd = PulsarSelectDataset()
        psm = MlpModel("pulsar_select_model", psd, [4])
        psm.exec_all()
    elif args.mode == 4:
        fd = FlowersDataset()
        fm = MlpModel("flowers_model_1", fd, [10])
        fm.exec_all(epoch_count=10, report=2)
    elif args.mode == 5:
        fd = FlowersDataset()
        fm2 = MlpModel("flowers_model_2", fd, [30, 10])
        fm2.exec_all(epoch_count=10, report=2)
