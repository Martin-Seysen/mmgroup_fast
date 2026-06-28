
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastG
from mmgroup import MM0, MMV, MM


MMV3 = MMV(3)


@pytest.mark.mmm
@pytest.mark.mm_op
def test_basis():
    for i in range(1):
        a = MMOpFastG()
        print("aug1", hex(a._augment(7, 0x8)))
        #print("aug2", hex(a._augment(8)))
        #print(a)
        del a