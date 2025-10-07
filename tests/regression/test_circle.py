from prt.datasets.pytorch.circle import CircleDataset


def test_circle_shape():
    dataset = CircleDataset()
    assert len(dataset) == 1000
    assert dataset.x.shape == (1000,)
    assert dataset.y.shape == (1000, 2)
