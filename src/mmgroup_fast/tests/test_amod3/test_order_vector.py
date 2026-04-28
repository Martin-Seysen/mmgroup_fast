import os

import numpy as np
import pytest

from mmgroup import MMV, MM0, MM, MMSpace, Xsp2_Co1, XLeech2
from mmgroup_fast.mm_op_fast import MMOpFastMatrix
from mmgroup_fast.dev.mm_op_axis_mod3.start_vector_59 import read_vector_59_mod3


MMV3 = MMV(3)


@pytest.mark.mm_amod3
def test_order_vector():
    m = MMOpFastMatrix(3, 4, 1)
    m.set_vstd(hash=1)
    ov = m.row_as_mmv(3)
    ov_ref = read_vector_59_mod3()[2]
    #for tag in "ABCTXZY":
    #    print(tag)
    #    assert (ov[tag] == ov_ref[tag]).all()
    assert ov == ov_ref
