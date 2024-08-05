"""
@author: rongzhao
"""
import torch
import torch.nn.functional as F
import numpy as np


def get_indices(X_shape, DF, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.

        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.

        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_D, n_H, n_W = X_shape

    # get output size
    out_d = int((n_D + 2 * pad - DF) / stride) + 1
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(DF), HF * WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_d), out_h * out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1[:, np.newaxis] + everyLevels[
        np.newaxis]  # level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Level 1 vector.
    depth1 = np.tile(np.repeat(np.arange(HF), WF), DF)
    # Duplicate for the other channels.
    depth1 = np.tile(depth1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.tile(np.repeat(np.arange(out_h), out_w), out_d)
    # Create matrix of index i at every levels for each channel.
    # j = depth1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    j = depth1[:, np.newaxis] + everyLevels[np.newaxis]

    # ----Compute matrix of index k----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), DF * HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_d * out_h)
    # Create matrix of index j at every slides for each channel.
    # k = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    k = slide1[:, np.newaxis] + everySlides[np.newaxis]

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), DF * HF * WF).reshape(-1, 1)

    return i, j, k, d


def triplet(s):
    if isinstance(s, int):
        s = (s,) * 3
    return s


def im2col_loop(X, DF, HF, WF, stride, pad):
    n, c, d, h, w = X.shape
    s = triplet(stride)
    p = triplet(pad)
    newd = (d + 2 * p[0] - DF) // s[0] + 1
    newh = (h + 2 * p[1] - HF) // s[1] + 1
    neww = (w + 2 * p[2] - WF) // s[2] + 1
    if isinstance(X, torch.Tensor):
        data = F.pad(X, (p[0], p[0], p[1], p[1], p[2], p[2]))
        cols = torch.zeros((n, newd, newh, neww, np.prod([DF, HF, WF]) * c), dtype=torch.float32,
                           device=X.device)
    else:
        data = np.pad(X, pad_width=((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1]), (p[2], p[2])))
        cols = np.zeros((n, newd, newh, neww, np.prod([DF, HF, WF]) * c), dtype='float32')

    for i in range(newd):
        for j in range(newh):
            for k in range(neww):
                cols[:, i, j, k] = \
                    data[:, :, i * s[0]:i * s[0] + DF,
                    j * s[1]:j * s[1] + HF, k * s[2]:k * s[2] + WF].reshape(n, -1)
    if isinstance(X, torch.Tensor):
        cols = cols.permute(4, 0, 1, 2, 3).reshape(np.prod([DF, HF, WF]) * c, -1)
    else:
        cols = cols.transpose(4, 0, 1, 2, 3).reshape(np.prod([DF, HF, WF]) * c, -1)
    return cols


def im2col(X, DF, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    if isinstance(X, torch.Tensor):
        X_padded = F.pad(X, (pad,) * 6)
    else:
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad), (pad, pad)), mode='constant')
    i, j, k, d = get_indices(X.shape, DF, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j, k]
    print(d.max())
    print(i.max())
    print(j.max())
    print(k.max())
    if isinstance(X, torch.Tensor):
        cols = torch.cat([c for c in cols], dim=-1)
    else:
        cols = np.concatenate(cols, axis=-1)
    print(cols.shape)
    print(cols.reshape(-1))
    print('i', i.reshape(-1))
    print('j', j.reshape(-1))
    print('k', k.reshape(-1))
    return cols


def conv(X, weight, bias=None, stride=1, padding=1):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        Returns:
        - out: previous layer convolved.
    """
    m, n_C_prev, n_D_prev, n_H_prev, n_W_prev = X.shape

    n_C, _, DF, HF, WF = weight.shape

    n_D = int((n_D_prev + 2 * padding - DF) / stride) + 1
    n_H = int((n_H_prev + 2 * padding - HF) / stride) + 1
    n_W = int((n_W_prev + 2 * padding - WF) / stride) + 1

    X_col = im2col_loop(X, DF, HF, WF, stride, padding)  # C1k x NDHW
    w_col = weight.reshape((n_C, -1))  # C2 x C1k
    # Perform matrix multiplication.
    out = w_col @ X_col  # C2 x NDHW
    if bias is not None:
        bias = bias.reshape(-1, 1)
        out += bias
    # Reshape back matrix to image.
    if isinstance(X, torch.Tensor):
        out = torch.cat(torch.split(out, n_D * n_H * n_W, dim=-1), dim=0).reshape(
            (m, n_C, n_D, n_H, n_W))
    else:
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_D, n_H, n_W))
    return out


@torch.no_grad()
def quadra_solver(X, Y, G, rho, kD=3, kH=3, kW=3, stride=1, padding=1):
    c1 = X.shape[1]
    c2 = Y.shape[1]
    c1k = np.prod([c1, kD, kH, kW])
    x_col = im2col_loop(X.cpu().numpy(), kD, kH, kW, stride, padding)
    y = torch.cat([c for c in Y], dim=-1).reshape(c2, -1)  # c2, ndhw
    y = y.cpu().numpy()
    A = x_col @ x_col.T + rho * np.eye(c1k, dtype=x_col.dtype)
    B = y @ x_col.T + rho * G.reshape(c2, -1).cpu().numpy()
    w_star = 2 * B @ np.linalg.inv(A + A.T)
    w_star = w_star.reshape(c2, c1, 3, 3, 3)
    w_star = torch.from_numpy(w_star).float().to(X.device)
    return w_star


class QuadraSolver(object):
    def __init__(self, X, Y, kD, kH, kW, stride, padding, device, mu=0, eta=0, W0=None, att=None,
                 b0=None, eta_singular=True):
        """

        :param X: input -- Qact
        :param Y: target -- output_fp
        :param kD: kernel size along Depth
        :param kH: kernel size along Height
        :param kW: kernel size along Width
        :param stride: convolution stride
        :param padding: convolution padding
        :param device: device
        :param mu: scale of || W ||_2 term
        # :param eta: scale of || W - W0 ||_2 term
        :param W0: the FP weight
        :param att: an 'importance/attention' map on Y
        """
        self.kD = kD
        self.kH = kH
        self.kW = kW
        self.stride = stride
        self.padding = padding
        self.device = device
        self.mu = mu  # for || W ||_2 term
        # self.eta = eta  # for || W - W_0 ||_2 term
        self.dtype = 'float32'
        self.singular = False
        self.rho_modifier = 1
        self.att = att
        self.is_bias = b0 is not None
        # self.eta_singular = eta_singular
        if W0 is not None:
            self.W0 = W0.reshape(W0.shape[0], -1).to(device)  # c2 x c1k
        n = X.shape[0]
        c1 = X.shape[1]
        c2 = Y.shape[1]
        c1k = np.prod([c1, kD, kH, kW])
        self.n = n
        self.c1 = c1
        self.c2 = c2
        self.c1k = c1k
        self.eye = torch.eye(c1k, device=device)
        if self.is_bias:  # one more dimension for bias
            self.c1k += 1
            # print(f'W0 is {W0.shape}, b0 is {b0.unsqueeze(1).shape}')
            self.W0 = torch.cat([self.W0, b0.unsqueeze(1)], dim=1)  # c2 x c1k'
            self.eye = torch.eye(self.c1k, device=device)
            self.quasi_eye = torch.eye(self.c1k, device=device)
            self.quasi_eye[-1][-1].zero_()

        # execute im2col in numpy
        x_col = im2col_loop(X.cpu().numpy().astype(self.dtype), kD, kH, kW, stride,
                            padding)  # c1k x ndhw
        if self.is_bias:  # c1k' x ndhw
            x_col = np.concatenate((x_col, np.ones((1, x_col.shape[1]), dtype=self.dtype)), axis=0)
        x_col = torch.from_numpy(x_col.astype('float32'))

        self.oom = False
        try:
            x_col = x_col.to(device)
        except RuntimeError as e:
            self.oom = True
            print(e)
            # x_col = x_col
        if self.oom:
            y = torch.cat([c for c in Y.cpu()], dim=1).reshape(c2, -1)  # c2 x ndhw
        else:
            y = torch.cat([c for c in Y], dim=1).reshape(c2, -1).to(device)

        try:
            self.A0, self.B0 = self.getA0B0(x_col, y)
        except RuntimeError as e:
            self.oom = True
            print(e)
            torch.cuda.empty_cache()
            self.A0, self.B0 = self.getA0B0(x_col.cpu(), y.cpu())

        del x_col, y
        torch.cuda.empty_cache()

    def getA0B0(self, x_col, y):
        """
        Compute the invariant quantities over ADMM iters.
        :param x_col: columnized input (c1k x ndhw)
        :param y: target output (c2 x ndhw)
        :return: A0, B0
        """
        device = self.device
        n = self.n
        # with attention map
        if self.att is not None:
            if self.att.dim() == 4:  # pixel-wise importance, NDHW
                # print(f'x_col shape {x_col.shape} vs att shape {self.att.reshape(1, -1).shape}')
                x_colH = x_col * self.att.reshape(1, -1).to(x_col.device)  # broadcast: c1k x NDHW * 1 x NDHW
                # x_colH = x_colH.to(device)
            else:
                raise NotImplementedError
        else:
            x_colH = x_col

        A_tmp = 0
        B_tmp = 0
        dhw = x_col.shape[1] // n
        for i in range(n):
            xx = x_col[:, i * dhw:(i + 1) * dhw]  # c1k x dhw
            xxH = x_colH[:, i * dhw:(i + 1) * dhw]  # c1k x dhw
            yy = y[:, i * dhw:(i + 1) * dhw]  # c2 x dhw
            A_tmp += xx @ xxH.T  # c1k x c1k
            B_tmp += yy @ xxH.T  # c2 x c1k
        A0 = 2 * A_tmp.to(device)  # c1k x c1k
        B0 = (2 * B_tmp).to(device)  # c2 x c1k

        return A0, B0

    def getAB(self, rho, eta, G):
        if self.is_bias:
            A = self.A0 + (rho + self.mu) * self.quasi_eye + eta * self.eye
            B = self.B0 + eta * self.W0
            B[:, :self.c1k-1] += rho * G.reshape(self.c2, -1)
        else:
            A = self.A0 + (rho + self.mu + eta) * self.eye
            B = self.B0 + rho * G.reshape(self.c2, -1) + eta * self.W0  # .astype(self.dtype)
        # print(f'Cond. of A0 is {torch.linalg.cond(self.A0):.3f}, Cond. of A is {torch.linalg.cond(A):.3f}')
        return A, B

    def solve(self, rho, eta, G):
        A, B = self.getAB(rho, eta, G)
        if not self.oom:
            try:
                w_star = torch.linalg.solve(A, B.T).T  # solve AW.T = B.T
            except RuntimeError as e:
                self.oom = True
                print(e)
        if self.oom:
            w_star = torch.linalg.solve(A.cpu(), B.cpu().T).T
            w_star = w_star.to(self.device)

        if self.is_bias:
            b_star = w_star[:,-1]  # (c2, )
            w_star = w_star[:,:-1].reshape(self.c2, self.c1, self.kD, self.kH, self.kW)  # c2 x ...
            return w_star, b_star
        else:
            w_star = w_star.reshape(self.c2, self.c1, self.kD, self.kH, self.kW)
            return w_star


if __name__ == '__main__':
    inChan = 28
    outChan = 32
    N = 2
    D, H, W = 132, 32, 32
    k = 3
    stride = 2
    padding = 1
    bias = None

    img = torch.randn(N, inChan, D, H, W)
    weight = torch.randn(outChan, inChan, k, k, k)

    out1 = F.conv3d(img, weight, bias, stride, padding)
    out2 = conv(img, weight, bias, stride, padding)

    loss = F.mse_loss(out1, out2)
    print(loss)
