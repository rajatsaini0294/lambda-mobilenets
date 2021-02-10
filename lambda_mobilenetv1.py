import torch
import torch.nn as nn
import torch.nn.functional as F

from lambda_networks import LambdaLayer


class Identity(nn.Module):
    """ Identity layer to mimic no layer """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LambdaMobileNetV1(nn.Module):
    def __init__(self, num_classes: int):
        super(LambdaMobileNetV1, self).__init__()

        def conv_bn(inp: torch.Tensor, oup: torch.Tensor, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(
            inp: torch.Tensor, oup: torch.Tensor, stride: int, layer_type: str
        ) -> nn.Sequential:
            if layer_type == "c":

                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            elif layer_type == "l":
                return LambdaLayer(
                    dim=inp,
                    dim_out=oup,
                    r=3,
                    dim_k=4,
                    heads=4,
                    dim_u=1,
                )
            else:
                return Identity()

        self.layer_type = [
            "c",  # layer: 0
            "c",  # layer: 1
            "c",  # layer: 2
            "c",  # layer: 3
            "c",  # layer: 4
            "c",  # layer: 5
            "l",  # layer: 6
            "i",  # layer: 7
            "i",  # layer: 8
            "i",  # layer: 9
            "i",  # layer: 10
            "i",  # layer: 11
            "c",  # layer: 12
            "i",  # layer: 13
            "c",  # layer: 14
        ]  # c: conv_dw, l: Lambda layer, i: identity (no layer)

        self.model = nn.Sequential(
            conv_bn(
                3, 32, 1
            ),  # layer: 0 (remove stride 2 here to get better performance for cifar10/100)
            conv_dw(32, 64, 1, self.layer_type[1]),  # layer: 1
            conv_dw(64, 128, 2, self.layer_type[2]),  # layer: 2
            conv_dw(128, 128, 1, self.layer_type[3]),  # layer: 3
            conv_dw(128, 256, 2, self.layer_type[4]),  # layer: 4
            conv_dw(256, 256, 1, self.layer_type[5]),  # layer: 5
            conv_dw(256, 512, 2, self.layer_type[6]),  # layer: 6
            conv_dw(512, 512, 1, self.layer_type[7]),  # layer: 7
            conv_dw(512, 512, 1, self.layer_type[8]),  # layer: 8
            conv_dw(512, 512, 1, self.layer_type[9]),  # layer: 9
            conv_dw(512, 512, 1, self.layer_type[10]),  # layer: 10
            conv_dw(512, 512, 1, self.layer_type[11]),  # layer: 11
            conv_dw(512, 1024, 2, self.layer_type[12]),  # layer: 12
            conv_dw(1024, 1024, 1, self.layer_type[13]),  # layer: 13
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.fc = nn.Linear(1024, num_classes)  # layer: 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def get_mobilenetv1(num_classes: int) -> LambdaMobileNetV1:
    """ return the mobilenet-v1 model """

    return LambdaMobileNetV1(num_classes=num_classes)
