

class DQN(Module):
    def __init__(self, h, w, outputs, in_size=2):
        super(DQN, self).__init__()
        print(in_size)
        self.stack = CnvStack(in_size)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        linear_input_size = convw * convh * 16
        # print(linear_input_size, outputs)
        self.head1 = nn.Linear(linear_input_size, outputs)
        # self.head2 = nn.Linear(linear_input_size // 2, outputs)
        # self.feats = FeatureNet(linear_input_size // 2, 64, in_size)

    def forward(self, x, y=None):
        """Called with either one element to determine next action, or a batch
            during optimization. Returns tensor([[left0exp,right0exp]...])."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head1(x.view(x.size(0), -1))
        return self.bnO(x)