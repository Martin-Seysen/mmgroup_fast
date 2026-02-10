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

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))


from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import parse_mat24_orbits
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import axis_count_BpmC

try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False







#########################################################################
# convert axis to intance of class MMOpFastMatrix
#########################################################################

def axis_to_MMOpFastMatrix(axis):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0)
    return amod3



#########################################################################
# Check case 8B
#########################################################################


def iter_testcases_8B(n_cases = 1000):
    std_axis_8B = Axis.representatives()["8B"]
    for i in range(n_cases):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis_8B * g



def check_case_8B(n_cases = 200, verbose = 0):
    print("Testing reduction of axis of type 8B")
    if use_mmgroup_fast:
        n_tests, n_ops = 0, 0
        for n, axis in enumerate(iter_testcases_8B(n_cases)):
            if verbose:
                print("Test", n+1)
            amod3 = axis_to_MMOpFastMatrix(axis)
            ax_type, v = amod3.find_v4()
            assert ax_type == "8B", (ax_type, hex(v))
            if verbose > 1:
                print("v=", hex(v))
                print(amod3.data[1])
            ax1 = axis * Xsp2_Co1('c', v & 0xffffff) **-1
            assert '4A' in [ax1.axis_type(e) for e in range(1,3)]
            t_markers = axis_count_BpmC(ax1)
            if verbose:
                print("  Markers:", t_markers)
            assert set(t_markers) == {0, 78}
            e = t_markers.index(78) + 1
            assert ax1.axis_type(e) == '4A', ax1.axis_type(e)
            n_tests += 1
            n_ops += amod3.raw_data_flat_uint32[24*8+7]
        print("passed, %.2f operations mod 3 per test" % (n_ops/n_tests))
  
#########################################################################
# Main program
#########################################################################

test_8B_all = check_case_8B

if __name__ == "__main__":
    test_8B_all(n_cases = 400, verbose=0)

