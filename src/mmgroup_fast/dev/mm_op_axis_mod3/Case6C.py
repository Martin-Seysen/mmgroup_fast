import sys
import os
import re
from collections import defaultdict, OrderedDict
from functools import reduce
import operator
import numpy as np

from mmgroup import mat24, GcVector, GCode, AutPL, Cocode, PLoop
from mmgroup import MM, XLeech2, Xsp2_Co1, Octad, SubOctad
from mmgroup.bitfunctions import lin_table
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
from mmgroup.axes import Axis, BabyAxis
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
# Tables for code generation
#########################################################################

 #  The standard cocode vector [2,3] in Leech lattice encoding
STD_COC = 0x200

E6_ENTRIES = [0, 4, 5, 6, 7]


_E6_basis = None

def E6_basis():
    """Return basis of of space E6 of the Leech lattice mod 2

    This is broken!!!!!!!!!!!!!!!

    There is a subspace E8 of dimension 8 of the Leech lattice mod 2
    related to involution Y, which inverts the first 8 basis vectors
    of the Leech lattice. One element of E8 is standard cocode vector
    STD_COC defined above. The function returns a basis of the
    orthogonal complement E6 of STD_COC in E8 in Leech lattice
    encoding.

    The function returns a fixed basis of E6.
    """
    global _E6_basis
    if _E6_basis is None:
        X_VALUES = [4,5,6,7]
        _E6_basis = [XLeech2(0, Cocode([1, 2, 3, x])).ord for x in X_VALUES]
        _E6_basis += [0x800000, XLeech2(PLoop(range(8)), 0).ord]
    return _E6_basis



_E6_vectors_type4 = None

def E6_vectors_type4():
    """List of type-4 vectors in the subspace E6 of Leech lattice mod 2

    The function returns the list of these type-4 vectors E6 in
    co-ordinates of the basis returnd by function E6_basis().
    """
    global _E6_vectors_type4
    if _E6_vectors_type4 is None:
        _E6_vectors_type4  = [n for n, v
             in enumerate(lin_table(E6_basis()))
             if gen_leech2_type(v) == 4]
    return _E6_vectors_type4



def Y_E6():
    TAB = [0x200, 0x300]
    COCODES = [Cocode(c) for c in [[1,2], [1,3]]]
    GCODES =  [GCode(x) for x in TAB]
    for i, gc in enumerate(GCODES):
        for j, cc in  enumerate(COCODES):
            assert int(gc & cc) == (i ^ j ^ 1), (i, j)
    TAG_ATOM_Y = 0x40000000 
    return [TAG_ATOM_Y + x for x in lin_table(TAB)]

# print([hex(x) for x in  Y_E6()])



class Tables:
    directives = {}
    def __init__(self):
        self.tables = {
            "MM_AXIS3_CASE_6C_BASIS_E6": E6_basis(),
            "MM_AXIS3_CASE_6C_ENTRIES_E6": E6_ENTRIES,
            "MM_AXIS3_CASE_6C_TYPE4": E6_vectors_type4(),
            "MM_AXIS3_CASE_6C_TABLE_Y": Y_E6(),
        }

class MockupTables:
    directives = {}
    tables = defaultdict(lambda x: [0])



#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################

def axis_to_MMOpFastMatrix(axis):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0)
    return amod3






#########################################################################
# Check case 6C
#########################################################################

def iter_testcases_6C(n_cases = 100):
    std_axis_6C = Axis.representatives()["6C"]
    for i in range(n_cases):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis_6C * g


A_EXPECTED = np.array(
[[1, 0, 0, 0, 0, 0],
 [0, 0, 2, 2, 0, 0],
 [0, 2, 0, 2, 0, 0],
 [0, 2, 2, 0, 0, 0],
 [0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 1]], dtype = np.uint8)
 


def check_case_6C(n_cases = 100, verbose = 0):
    if use_mmgroup_fast:
        print("Testing axis type 6C")
        for n, axis in enumerate(iter_testcases_6C(n_cases)):
            if verbose:
                print("Test", n+1)
            amod3a = axis_to_MMOpFastMatrix(axis)
            buf = amod3a.analyze_v4()
            if verbose > 1:
                print(amod3a.data[1])
            assert (amod3a.data[1, :6, :6] == A_EXPECTED).all()
                
            amod3 = axis_to_MMOpFastMatrix(axis)
            ax_type, v = amod3.find_v4()
            assert ax_type == "6C", (ax_type, hex(v))
            g = Xsp2_Co1('c', v) **-1
            ax1 = axis * g
            t_markers = axis_count_BpmC(ax1)
            if verbose:
                ax1.display_sym()
                print(t_markers) 
            assert set(t_markers) == {3, 171}
            e = t_markers.index(171) + 1
            assert ax1.axis_type(e) == '4A', ax1.axis_type(e)


  
#########################################################################
# Main program
#########################################################################

def test_6C_all(n_tests = 100):
    check_case_6C(n_tests, verbose = 0) 

if __name__ == "__main__":
    test_6C_all()

