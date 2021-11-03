import numpy as np
import paddle
from reprod_log import ReprodLogger

from model.AlexNet import alexnet

if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = alexnet(True)
    model.eval()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)
    print(paddle.argmax(out, axis=1))
    # print(out.numpy())
    # s
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
