
import numpy as np
from random import randint, shuffle

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastG
from mmgroup import MM0, MMV, MM


MMV3 = MMV(3)


@pytest.mark.mmm
@pytest.mark.mm_op
def test_basis():
    for i in range(10):
        a = MMOpFastG()
        """
        # The following test produces an error!!!
        a._display_flags()
        print(a)
        """
        del a