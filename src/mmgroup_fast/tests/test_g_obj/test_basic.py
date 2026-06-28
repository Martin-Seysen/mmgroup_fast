
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
        a0 = MM0('r')
        a = MMOpFastG(a0)
        print("\nA test element is: ", a)
        #assert MM(a0) == MM('a', a.mmdata) # This fails!!!
        del a