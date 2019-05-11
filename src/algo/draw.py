import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable


def unit_prefix(x, n=1):
    for i in range(n):
        x = x.unsqueeze(0)
    return x


def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd:
        y = unit_prefix(y, xd - yd)
    elif yd > xd:
        x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if ys[td] == 1:
            ys[td] = xs[td]
        elif xs[td] == 1:
            xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)


def matmul(X, Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i], Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)


class DrawModel(nn.Module):
    def __init__(self, T, A, B, z_size, N, dec_size, enc_size):
        """

        :param T: time steps
        :param A: img_
        :param B:
        :param z_size:
        :param N:
        :param dec_size:
        :param enc_size:
        """
        super(DrawModel, self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.batch_size = 64
        self.logsigmas, self.sigmas, self.mus = [0] * T, [0] * T, [0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size, dec_size)
        self.decoder_gru = nn.GRUCell(z_size, dec_size)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.dec_w_linear = nn.Linear(dec_size, N*N)

        self.sigmoid = nn.Sigmoid()

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size)
        mu = self.mu_linear(h_enc)              # 1
        log_sigma = self.sigma_linear(h_enc)    # 2
        sigma = torch.exp(log_sigma)
        return mu + sigma * e, mu, log_sigma, sigma

    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = align(tmp, g)
        mu = tmp_t + g_t
        return mu

    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a, t_mu_x = align(a, mu_x)
        temp = t_a - t_mu_x
        temp, t_sigma = align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        f_ = torch.exp(-torch.pow(temp, 2))
        f_ = f_ / (f_.sum(2, True).expand_as(f_) + epsilon)
        return f_

    def filterbank(self, gx, gy, sigma2, delta):
        rng = Variable(torch.arange(0, self.N).view(1, -1))
        mu_x = self.compute_mu(gx, rng, delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = torch.arange(0, self.A).view(1, 1, -1)
        b = torch.arange(0, self.B).view(1, 1, -1)

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = self.filterbank_matrices(a, mu_x, sigma2)
        Fy = self.filterbank_matrices(b, mu_y, sigma2)
        return Fx, Fy

    def forward(self, x):
        self.batch_size = x.size()[0]
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size)
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size)

        enc_state = torch.zeros(self.batch_size, self.enc_size)
        dec_state = torch.zeros(self.batch_size, self.dec_size)
        for t in range(self.T):
            # prev state
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t-1]

            # img - sigm(prev_state)
            x_hat = x - self.sigmoid(c_prev)            # 3
            r_t = self.read(x, x_hat, h_dec_prev)       # 4    read (image, diff, state)
            h_enc_prev, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state))  # 5

            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc_prev)    # 6

            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))     # 7

            self.cs[t] = c_prev + self.write(h_dec)                         # 8
            h_dec_prev = h_dec

    def loss(self, x):
        self.forward(x)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons, x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * self.T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * (torch.sum(mu_2 + sigma_2 - 2 * logsigma, 1) - self.T)
            Lz += kl_terms[t]

        Lz = torch.mean(Lz)
        loss = Lz + Lx    # 12
        return loss

    def attn_window(self, h_dec):
        """
        Args:
            h_dec

        compute the parameters for the attention grid
        """
        gx_, gy_, log_sigma_2, log_delta, log_gamma = self.dec_linear(h_dec).split(1, 1)  # 21

        # xy at center
        gx = ((self.A + 1) / 2) * (gx_ + 1)               # 22
        gy = ((self.B + 1) / 2) * (gy_ + 1)               # 23
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma

    def read(self, x, x_hat, h_dec_prev):
        """
        x: img
        x_hat
        """
        # attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

        def filter_img(img, Fx, Fy, gamma, A, B, N):
            """ """
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N*N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)
        return torch.cat((x, x_hat), 1)

    def write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size, self.N, self.N)

        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)

        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A*self.B)
        return wr / gamma.view(-1, 1).expand_as(wr)

    def generate(self, batch_size=64):
        h_dec_prev = Variable(torch.zeros(batch_size, self.dec_size), volatile=True)
        dec_state = Variable(torch.zeros(batch_size, self.dec_size), volatile=True)

        for t in range(self.T):
            c_prev = Variable(torch.zeros(batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs







