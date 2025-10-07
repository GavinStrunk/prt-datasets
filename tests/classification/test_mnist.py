import torch
import prt.datasets.pytorch.mnist as dpm


def test_setup():
    assert True


def test_mnist_shapes():
    data = dpm.MNIST(download=True)

    assert len(data) == 60000
    assert isinstance(data[0], tuple)
    assert data[0][0].shape == (1, 28, 28)
    assert isinstance(data[0][1], int)


def test_mnist_train_shapes():
    data = dpm.MNIST(train=True, download=True)

    assert len(data) == 60000
    assert isinstance(data[0], tuple)
    assert data[0][0].dtype is torch.float32
    assert data[0][0].shape == (1, 28, 28)
    assert torch.max(data[0][0]) < 2.822
    assert torch.min(data[0][0]) > -0.425
    assert isinstance(data[0][1], int)


def test_mnist_value_exclude():
    data = dpm.MNIST(exclude_digits=[4])
    assert len(data) == 54158
    assert data.__len__() == 54158

    data = dpm.MNIST(exclude_digits=[5])
    assert len(data) == 54579


def test_mnist_list_exclude():
    data = dpm.MNIST(exclude_digits=[4, 5])
    assert len(data) == 48737
