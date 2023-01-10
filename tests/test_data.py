from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
import pytest
import torch
#dataset = CorruptMnist()
#dataset = MNIST(...)

#class TestClassCorruptMnist:

#    def test_train_len(self):
#        assert len(CorruptMnist(train=True)) == 5000


@pytest.mark.parametrize('train, expected_value', [(True, 40000), (False, 5000)])
def test_train_len(train, expected_value):
    dataset = CorruptMnist(train=train, in_folder="data/raw", out_folder="data/processed")
    assert len(dataset) == expected_value

@pytest.mark.parametrize('train', [True,False])
def test_shape_elements(train):
    dataset = CorruptMnist(train=train, in_folder="data/raw", out_folder="data/processed")
    print
    size = torch.Size([1,28,28])
    #assert dataset[0][0].shape == torch.Size([1,28,28])
    assert all([a.shape == size for a, b in dataset])

@pytest.mark.parametrize('train', [True,False])
def test_all_elements_present(train):
    dataset = CorruptMnist(train=train, in_folder="data/raw", out_folder="data/processed")
    sets = set(range(0,10))
    print(dataset[0][1].item())
    lists = set([b.item()  for a, b in dataset])
    assert sets == lists

##assert len(dataset) == N_train for training and N_test for test
#assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
#assert that all labels are represented
