import torch
import torch.nn as nn

class PVNet(nn.Module):
    def __init__(self, board_shape, action_shape):
        super(PVNet, self).__init__()
        self.NUM_STATE_LAYERS = board_shape[0]
        self.BOARD_SIZE = board_shape[1]
        self.NUM_ACTION_LAYERS = action_shape[0]
        self.ACTION_SHAPE = (
            self.BOARD_SIZE, self.BOARD_SIZE, self.NUM_ACTION_LAYERS)
        
        self.data_format = 'channels_first'
        self.framework = 'pytorch'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Common
        self.relu = nn.ReLU()

        # Conv block
        self.conv_conv = nn.Conv2d(self.NUM_STATE_LAYERS, 128, 3, padding=1)
        self.conv_bn   = nn.BatchNorm2d(num_features=128)

        # Residual block
        self.res_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.res_bn = nn.BatchNorm2d(num_features=128)

        # Policy head
        self.policy_conv = nn.Conv2d(128, self.NUM_ACTION_LAYERS, 1)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, 1)
        self.value_bn = nn.BatchNorm2d(num_features=1)
        self.value_fc1 = nn.Linear(self.BOARD_SIZE ** 2, 64)
        self.value_fc2 = nn.Linear(64, 2)


        self.to(self.device)


    def evaluate(self, board):
        prior, V = None, None
        # ------------------ TODO ------------------ #
        #                                            #
        # get prior, make V (vector) from v (scalar) #
        #                                            #
        # ------------------------------------------ #
        return prior, V

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[None]
        if not torch.is_tensor(x):
            x = torch.tensor(x.astype('float'), dtype=torch.float32).to(self.device)
        
        x = self.convolutional_block(x)
        for i in range(10):
            x = self.residual_block(x)
        prior = self.policy_head(x)
        V = self.value_head(x)

        return prior, V

    def convolutional_block(self, x):
        # output.shape == (None, 128, size, size)
        conv = self.conv_conv(x)
        bn   = self.conv_bn(conv)
        relu = self.relu(bn)
        return relu

    def residual_block(self, x):
        # input and output shapes are (None, 128, size, size)
        conv1 = self.res_conv(x)
        bn1   = self.res_bn(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.res_conv(relu1)
        bn2   = self.res_bn(conv2)
        skip  = x + bn2
        relu2 = self.relu(skip)
        return relu2

    def policy_head(self, x):
        conv   = self.policy_conv(x)
        shape  = conv.shape
        policy = nn.Softmax(dim=1)(conv.view(shape[0], -1)).view(shape)
        return policy

    def value_head(self, x):
        conv  = self.value_conv(x)
        bn    = self.value_bn(conv)
        relu1 = self.relu(bn).view(bn.shape[0], -1)
        fc1   = self.value_fc1(relu1)
        relu2 = self.relu(fc1)
        fc2   = self.value_fc2(relu2)
        V     = nn.Softmax(dim=1)(fc2) - 0.5
        return V


if __name__ == '__main__':
    import numpy as np
    from model import Blokus

    BOARD_SIZE = 13
    env = Blokus(BOARD_SIZE, [0, 1])
    net = PVNet(env.BOARD_SHAPE, env.ACTION_SHAPE)

    state = env.reset()
    import numpy as np; batch = np.random.randn(2,51,13,13)
    board = state.board
    net(batch)
