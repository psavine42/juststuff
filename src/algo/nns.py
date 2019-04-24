import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple
import numpy as np


def conv2d_size_out(size, k=5, s=2):
    return (size - (k - 1) - 1) // s + 1


def conv2d_size_outm(size, mod):
    return (size - (mod.kernel_size[0] - 1) - 1) // mod.stride[0] + 1

def convs2d_size_out(size, convs):
    res = size
    for conv in convs:
        res = conv2d_size_outm(res, conv)
    return res

class RLdraw(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


_map = {2: [[2, 16, 3, 2], [16, 32, 3, 2], [32, 32, 3, 1]],
        3: [[3, 16, 5, 2], [16, 32, 3, 1], [32, 32, 3, 1]],
        4: [[3, 16, 5, 2], [16, 32, 3, 1], [32, 32, 3, 1]]
        }


class FeatureNet(nn.Module):
    """ """
    def __init__(self, in_size, output_size):
        super(FeatureNet, self).__init__()
        self.linear1 = nn.Linear(in_size, 2 * in_size)
        self.linear2 = nn.Linear(2 * in_size, output_size)
        # self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        # print(x.size())
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class DQN(nn.Module):
    def __init__(self, h, w, outputs, in_size=2):
        super(DQN, self).__init__()
        print(in_size)
        if in_size == 2:
            self.conv1 = nn.Conv2d(in_size, 16, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(32)
            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, k=3, s=2), k=3, s=2), k=3, s=1)
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, k=3, s=2), k=3, s=2), k=3, s=1)
        else:
            self.conv1 = nn.Conv2d(in_size, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.bnO = nn.LayerNorm(outputs , elementwise_affine=False)
            convw = convs2d_size_out(w, [self.conv1, self.conv2, self.conv3])
            convh = convs2d_size_out(h, [self.conv1, self.conv2, self.conv3])
            # convw = conv2d_size_outm(conv2d_size_out(conv2d_size_outm(w), k=3, s=1), k=3, s=1)
            # convh = conv2d_size_outm(conv2d_size_out(conv2d_size_outm(h), k=3, s=1), k=3, s=1)

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


class Controller:
    def __init__(self):
        pass


class DQNC(nn.Module):
    def __init__(self, h, w, outputs, in_size=2):
        super(DQNC, self).__init__()
        print(in_size)
        if in_size == 2:
            self.conv1 = nn.Conv2d(in_size, 16, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(32)
            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, k=3, s=2), k=3, s=2), k=3, s=1)
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, k=3, s=2), k=3, s=2), k=3, s=1)
        else:
            self.conv1 = nn.Conv2d(in_size, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(16)
            self.bnO = nn.LayerNorm(outputs , elementwise_affine=False)
            convw = convs2d_size_out(w, [self.conv1, self.conv2, self.conv3])
            convh = convs2d_size_out(h, [self.conv1, self.conv2, self.conv3])
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        features_out = 40

        linear_input_size = convw * convh * 16 + features_out
        print('linear in size ', linear_input_size, outputs)
        self.head1 = nn.Linear(linear_input_size, linear_input_size // 2)
        self.head2 = nn.Linear(linear_input_size // 2, outputs)
        # self.advantage = nn.Linear(linear_input_size, 1)
        self.features = FeatureNet(11, features_out)
        # self.head2 = nn.Linear(linear_input_size // 2, outputs)
        # self.feats = FeatureNet(linear_input_size // 2, 64, in_size)

    def forward(self, inputs):
        """Called with either one element to determine next action, or a batch
            during optimization. Returns tensor([[left0exp,right0exp]...])."""
        # print(x.size())
        x, feats = inputs

        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        y = self.features(feats.view(-1).unsqueeze(0))
        # print(x.size(), y.size())
        xs = torch.cat((x, y), -1)
        xs = F.relu(self.head1(xs))
        xs = self.head2(xs)
        return F.softmax(xs, dim=1)


class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Canny(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(Canny, self).__init__()
        from scipy.signal import gaussian
        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD
        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        # with tf.variable_scope(scope):
            # Input and visual encoding layers
        self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
        self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
        hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

        # Recurrent network for temporal dependencies
        self.lstm_cell = nn.LSTMCell(256, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(self.imageIn)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 256])

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
        self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

