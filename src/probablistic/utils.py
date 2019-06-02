import torch
import torch.autograd


def flatten(input):
    if isinstance(input, (list, tuple)):
        return torch.cat([flatten(x) for x in input], -1)
    return input.view(input.size(0), -1)


def dfdx(f, x):
    """
    see https://discuss.pytorch.org/t/how-to-use-the-partial-derivative-of-an-op-as-new-op/12255/4
    :param f:
    :param x:
    :return: partial derivative f wrt x
    """
    # print('df', type(x), size(x))
    dfdx_val, = torch.autograd.grad(outputs=[f],
                                    inputs=[x],
                                    grad_outputs=None,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True,
                                    allow_unused=True
                                    )
    return dfdx_val


def size(tensors):
    if torch.is_tensor(tensors):
        return tensors.size()
    return [size(t) for t in tensors]


def scale01(tnsr, min=0, max=1):
    return (tnsr - tnsr.min()) / (tnsr.max() - tnsr.min())


def norm2(tnsr):
    return tnsr / tnsr.norm(2)

