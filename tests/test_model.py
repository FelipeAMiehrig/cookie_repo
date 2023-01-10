from tests import _PATH_DATA
from src.data.make_dataset import CorruptMnist
from src.models.model import MyAwesomeModel
import pytest
import torch
import random 

def test_model():
    N = int(random.random()*10000)
    input = torch.rand((N,1,28,28))
    output = MyAwesomeModel().forward(input)
    assert input.shape[0]==output.shape[0]
