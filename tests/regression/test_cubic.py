import pytest
import torch
import prt.datasets.pytorch.cubic as dpc


def test_setup():
    assert True


def test_cubic_training_set():
    dataset = dpc.CubicDataset(train=True)
    assert len(dataset) == 1000
    assert isinstance(dataset[0], tuple)
    assert dataset[0][0].shape == torch.Size([1,])
    assert dataset[0][1].shape == torch.Size([1,])

    # Test the range of the dataset
    x_min = float(dataset.x.min())
    x_max = float(dataset.x.max())
    assert x_min >= -4 and x_max <= 4


def test_cubic_test_set():
    dataset = dpc.CubicDataset(train=False)
    # Test the range of the dataset
    x_min = float(dataset.x.min())
    x_max = float(dataset.x.max())
    assert x_min >= -7 and x_max <= 7


def test_cubic_dataloader():
    dataset = dpc.CubicDataset(train=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for batch_data, batch_label in loader:
        assert batch_data.shape == torch.Size([10, 1])
        assert batch_label.shape == torch.Size([10, 1])
        break  # Testing just the first batch


def test_cubic_no_noise():
    dataset = dpc.CubicDataset(noise=False)
    # When noise is false, y should equal x^3
    x, y = dataset[0]
    assert torch.allclose(y, x ** 3), "Y should equal x^3 when noise is false"


def test_cubic_seed():
    seed = 10
    dataset1 = dpc.CubicDataset(seed=seed)
    dataset2 = dpc.CubicDataset(seed=seed)
    assert torch.equal(dataset1.x, dataset2.x), "X values should be equal with seed control"
    assert torch.equal(dataset1.y, dataset2.y), "Y values should be equal with seed control"


def test_cubic_sample_count():
    num_samples = 500
    dataset = dpc.CubicDataset(num_samples=num_samples)
    assert len(dataset) == num_samples, "Dataset should contain the specified number of samples"
