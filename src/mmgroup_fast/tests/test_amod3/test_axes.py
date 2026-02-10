from collections import OrderedDict
import numpy as np
from random import randint, shuffle, sample

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastMatrix, MMOpFastAmod3
from mmgroup import MM0, MMV, MM, XLeech2, Cocode
from mmgroup.axes import Axis


DIAG = [1, 0, 0]

@pytest.mark.mm_amod3
def test_axes_amod3():
    print()
    axis_reps = Axis.representatives()
    for orbit, axis in axis_reps.items():
        ax = axis  # * MM('r', 'G_x0')
        a = MMOpFastAmod3(ax.v15 % 3)
        norm = a.norm
        a.raw_echelon(DIAG[norm])
        d_img = a.dim_img
        if d_img == 24:
            a.raw_echelon(5)
        diag =  a.diag
        d_img = a.dim_img
        print(f"Orbit {orbit:>3}: {norm} {diag} {d_img:2}")
