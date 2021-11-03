# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Layer):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.input_grad = None
        self.conv_grad = []

        self.f1_conv = nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2)
        self.f2 = nn.ReLU()
        self.f3 = nn.MaxPool2D(kernel_size=3, stride=2)
        self.f4_conv = nn.Conv2D(64, 192, kernel_size=5, padding=2)
        self.f5 = nn.ReLU()
        self.f6 = nn.MaxPool2D(kernel_size=3, stride=2)
        self.f7_conv = nn.Conv2D(192, 384, kernel_size=3, padding=1)
        self.f8 = nn.ReLU()
        self.f9_conv = nn.Conv2D(384, 256, kernel_size=3, padding=1)
        self.f10 = nn.ReLU()
        self.f11_conv = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.f12 = nn.ReLU()
        self.f13 = nn.MaxPool2D(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))

        self.c1 = nn.Dropout(p=0)
        self.c2 = nn.Linear(256 * 6 * 6, 4096)
        self.c3 = nn.ReLU()
        self.c4 = nn.Linear(4096, 4096)
        self.c5 = nn.ReLU()
        self.c6 = nn.Linear(4096, num_classes)

    def forward(self, x):
        def _record_gradients(grad):
            self.input_grad = grad.detach().cpu().numpy()

        def _record_conv_gradients(grad):
            self.conv_grad.append(grad.detach().cpu().numpy())

        x.register_hook(_record_gradients)

        x = self.f1_conv(x)
        x.register_hook(_record_conv_gradients)

        x = self.f2(x)
        x = self.f3(x)
        x = self.f4_conv(x)
        x.register_hook(_record_conv_gradients)

        x = self.f5(x)
        x = self.f6(x)
        x = self.f7_conv(x)
        x.register_hook(_record_conv_gradients)

        x = self.f8(x)
        x = self.f9_conv(x)
        x.register_hook(_record_conv_gradients)

        x = self.f10(x)
        x = self.f11_conv(x)
        x.register_hook(_record_conv_gradients)

        x = self.f12(x)
        x = self.f13(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)

        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)

        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = paddle.load("weights/alexnet-paddlerename.pdparams")
        model.set_dict(state_dict)
    return model
