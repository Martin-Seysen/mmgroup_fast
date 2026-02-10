import sys
import os
import re
from collections import defaultdict, OrderedDict
import numpy as np
from mmgroup import MM, XLeech2, mat24, GcVector, GCode, AutPL, Xsp2_Co1
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_subtype, gen_leech3_op_vector_word
from mmgroup.generators import gen_leech3_op_vector_atom
from mmgroup.generators import gen_leech3_reduce, gen_leech3to2_short
from mmgroup.generators import gen_leech3_add, gen_leech2to3_abs, gen_leech3_neg
from mmgroup.generators import gen_leech3_reduce_leech_mod3, gen_leech3_neg
from mmgroup.generators import gen_leech2_op_word
from mmgroup.generators import gen_leech2_op_word_leech2_many
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_echelon_eqn
from mmgroup.clifford12 import leech2_matrix_orthogonal
from mmgroup.clifford12 import leech2_matrix_radical
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_subspace_eqn
from mmgroup.clifford12 import bitmatrix64_vmul
from mmgroup.axes import Axis
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))


from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import parse_mat24_orbits
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import py_prep_fixed_leech2_set
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import axis_count_BpmC

try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_prep_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_map_Case6A
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_find_case_2A
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False








#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################

def axis_to_MMOpFastMatrix(axis):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0)
    return amod3



#########################################################################
# Check case 6F
#########################################################################


def iter_testcases_6F(n_tests = 100):
    std_axis_6f = Axis.representatives()["6F"]
    for i in range(n_tests):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis_6f * g



def check_case_6F(n_tests = 100, verbose = 0):
    if use_mmgroup_fast:
        for axis in iter_testcases_6F(n_tests):
            amod3a = axis_to_MMOpFastMatrix(axis)
            buf = amod3a.analyze_v4()
            if verbose:
                print(amod3a.data[1])
                print([hex(x) for x in buf])
            t, v = amod3a.find_v4()
            assert t == "6F", (t, hex(v))
            ax1 = axis * Xsp2_Co1('c', v) **-1
            t_markers = axis_count_BpmC(ax1)
            #print(t_markers)
            assert set(t_markers) == {0, 8}
            e = t_markers.index(8) + 1
            assert ax1.axis_type(e) == '4C', ax1.axis_type(e)

  
#########################################################################
# Main program
#########################################################################

test_6F_all = check_case_6F

if __name__ == "__main__":
    test_6F_all()

