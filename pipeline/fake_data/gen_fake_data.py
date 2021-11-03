import numpy as np


def gen_fake_data():
    fake_data = np.random.rand(1, 3, 227, 227).astype(np.float32) - 0.5
    fake_label = (np.random.rand(1, 10) * 10).astype(np.int64)
    print(fake_label.shape)
    np.save("fake_data.npy", fake_data)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()
