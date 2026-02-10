import numpy as np
from mmgroup.mat24 import mul_perm, inv_perm
from random import shuffle, randint
from mmgroup_fast.display import mmgroup_fast_test_shuffle64_16
from mmgroup_fast.mm_op_fast import mmgroup_fast_test_shuffle64
import pytest



def ref_shuffle64(data, mask):
    return [data[mask[i]] for i in range(64)]








@pytest.mark.mm_op
def test_shuffle_64(verbose = 0):
    for i in range(1000):
        mask = list(range(64))
        shuffle(mask)
        mask = np.array(mask, dtype = np.uint8)
        data = [randint(0,255) for x in range(64)]
        data = np.array(data, dtype = np.uint8)
        shuffled = np.zeros(64, dtype = np.uint8)
        mmgroup_fast_test_shuffle64(data, mask, shuffled)
        shuffled = list(shuffled)
        ref_shuffled = ref_shuffle64(data, mask)
        if verbose:
            print(data)
            print(mask)
            print(np.array(shuffled, dtype = np.uint8))
            print(np.array(ref_shuffled, dtype = np.uint8))
        assert ref_shuffled == shuffled
         





