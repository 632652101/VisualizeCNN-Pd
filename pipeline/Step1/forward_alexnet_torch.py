import numpy as np
import torch
from reprod_log import ReprodLogger

from model.AlexNet_torch import alexnet


if __name__ == '__main__':
    reprod_logger = ReprodLogger()

    model = alexnet(True)
    model.eval()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)
    # forward
    out = model(fake_data)
    print(torch.argmax(out, dim=1))
    # print(out.detach().numpy())
    #
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
