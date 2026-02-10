import pytest

from mmgroup.generators import mm_group_invert_word
from mmgroup_fast.mm_op_fast import MMOpFastMatrix, MMOpFastAmod3
from mmgroup import MM0, MMV, MM, XLeech2, Cocode, Xsp2_Co1
from mmgroup.axes import Axis





@pytest.mark.mm_amod3
def test_mmgroup_fast_find_v4(n_tests = 50):
    from mmgroup_fast.dev.mm_op_axis_mod3.do_test_all import test_all
    test_all(n_tests)


