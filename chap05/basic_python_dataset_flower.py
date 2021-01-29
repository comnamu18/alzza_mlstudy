from basic_python_dataset import Dataset
from basic_python_mathutil import *  # noqa


class FlowersDataset(Dataset):
    def __init__(self, resolution=[100, 100], input_shape=[-1]):
        super(FlowersDataset, self).__init__("flowers", "select")

        path = "flowers"
        self.target_names = list_dir(path)

        images = []
        idxs = []

        for dx, dname in enumerate(self.target_names):
            subpath = path + "/" + dname
            filenames = list_dir(subpath)
            for fname in filenames:
                if fname[-4:] != ".jpg":
                    continue
                imagepath = os.path.join(subpath, fname)
                pixels = load_image_pixels(imagepath, resolution, input_shape)
                images.append(pixels)
                idxs.append(dx)

        self.image_shape = resolution + [3]

        xs = np.asarray(images, np.float32)
        ys = onehot(idxs, len(self.target_names))

        self.shuffle_data(xs, ys, 0.8)

    def visualize(self, xs, estimates, answers):
        draw_images_horz(xs, self.image_shape)
        show_select_results(estimates, answers, self.target_names)
