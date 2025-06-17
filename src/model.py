import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, num_input=17, num_filters=128):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input, num_filters, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        return self.conv1_act(y)


class ResBlock(nn.Module):

    def __init__(self, num_filters=128):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


class EvalBlock(nn.Module):

    def __init__(self, num_input=128, num_filters=16):
        super().__init__()
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(num_input, num_filters, 1, 1, 0)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_filters * 8 * 8, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_act = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        return self.tanh(x)


class MoveBlock(nn.Module):

    def __init__(self, num_input=128, num_filters=32):
        super().__init__()
        self.num_filters = num_filters
        self.conv1 = nn.Conv2d(num_input, num_filters, 1, 1, 0)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_filters * 8 * 8, 64 * 64)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_act(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.softmax(x)


class FENAnalyzerModel(nn.Module):

    def __init__(self, num_input_channels = 17, num_filter_channels=128, num_res_blocks=9):
        super().__init__()
        self.conv_block = ConvBlock(num_input=num_input_channels, num_filters=num_filter_channels)
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_filter_channels) for _ in range(num_res_blocks)])
        self.eval_block = EvalBlock(num_input=num_filter_channels)
        self.move_block = MoveBlock(num_input=num_filter_channels)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        value = self.eval_block(x)
        prob = self.move_block(x)
        return value, prob


if __name__ == "__main__":
    am = FENAnalyzerModel()
    input = torch.randn(64, 15, 8, 8)
    output = am(input)
    print("output",output.shape)
    print("output",output)
