

import sys
import warnings
import numpy as np

import pytest



from mmgroup_fast.display import mmgroup_fast_test_shuffle16



@pytest.mark.basic
def test_shuffle16():
    data = np.array(range(1,17), dtype = np.uint8)
    mask = np.array(range(15,-1,-1), dtype = np.uint8)
    res = mmgroup_fast_test_shuffle16(data, mask)
    ok = res < 0 or list(data) == list(range(16,0,-1))
    if not ok:
        print("test_shuffle16 returns", res, ", result is:")
        print(data)
        raise ValueError("Function mmgroup_fast_test_shuffle16 has failed")




 
       