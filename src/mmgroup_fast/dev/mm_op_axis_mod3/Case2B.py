import sys
import os
import re
from random import randint
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.linalg import hadamard

from mmgroup.bitfunctions import bit_mat_mul 
from mmgroup import MM, XLeech2, mat24, GcVector, GCode, AutPL, Xsp2_Co1
from mmgroup import Cocode, Octad, SubOctad, octad_entries
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_coarse_subtype
from mmgroup.generators import gen_leech3_op_vector_word
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
from mmgroup.mm_op import mm_op_eval_A
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.axes import Axis, BabyAxis




if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))





try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_prep_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_map_Case6A
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_find_case_2A
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import parse_mat24_orbits  
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import py_prep_fixed_leech2_set
    from mmgroup_fast.dev.case_2B.axis_dict import AXIS_DICT
    from mmgroup_fast.dev.case_2B.axis_class_2B_axes import AXIS_Y, NEG_AXIS
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE2BASIS
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE4IBASIS
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import COEFF
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE4I
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import SH_E9
    from mmgroup_fast.dev.case_2B.axis_class_2B import target_dicts
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import axis_count_BpmC
    STD_E8_TYPE2 = np.array([t[0] for t in SH_E9], dtype = np.uint32)
    assert len(STD_E8_TYPE2) == 120
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False
    raise



#########################################################################
# Standard list of type-2 vectors in standard octad subspace of Lambda
#########################################################################


def make_std_typ2_list():
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE2BASIS
    from mmgroup_fast.dev.case_2B.py_process_axis_2B import BASIS_E8
    STD_OCTAD = [0,1,2,3,4,5,6,7]
    data = []
    for x in TYPE2BASIS:
        data.append(int(bit_mat_mul(x, BASIS_E8)))
    # Next we check that the created list ``data`` is correct
    k = 0
    for i in range(8):
        for j in range(i):
            ref = Cocode([i,j]).ord
            assert data[k] == ref
            assert data[k + 28] == ref ^ 0x800000
            k += 1
    oct = Octad(STD_OCTAD)
    assert octad_entries(oct) == STD_OCTAD
    ref0 = SubOctad(oct, 0).ord
    for i in range(64):
        ref = SubOctad(oct, i).ord
        assert data[56+i] == ref
        cref = ref ^ ref0
        assert cref & 0xff7ff800 == 0
        coc = mat24.cocode_syndrome(cref, 0)
        if coc & 0x80:
            coc ^= 0xff
        assert coc == 2 * i + (i.bit_count() & 1)
    assert len(data) == 120 
    return data

_std_typ2_list = None

def std_typ2_list():
    global _std_typ2_list
    if _std_typ2_list is not None:
        return _std_typ2_list
    _std_typ2_list = make_std_typ2_list()
    return _std_typ2_list
    



#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################

def axis_to_MMOpFastMatrix(axis, baby_axis = None):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    if baby_axis is not None:
        m3.set_row(1, baby_axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0, -1 if baby_axis is None else 1)
    return amod3




#########################################################################
# Check cases 2B0 and 2B1
#########################################################################

STD_2A = XLeech2(0x200)


def iter_testcases_2Bx(ax_type, n_cases = 100):
    std_axis_2B = Axis.representatives()["2B"]
    for i in range(n_cases):
        axis =BabyAxis.representatives()["2B" + str(int(ax_type))]
        g = Xsp2_Co1('r', 'G_x0')
        ax = Axis(axis) * g
        yield ax, axis_to_MMOpFastMatrix(ax), STD_2A * g



def check_case_2Bx(n_cases = 100, verbose = 0):
    if use_mmgroup_fast:
        for ax_type in (0, 1):
            print("Checking baby axis case 2B%d" % ax_type) 
            for ax, amod3, v2 in iter_testcases_2Bx(ax_type, n_cases):
                amod3a = MMOpFastAmod3(amod3)
                buf = amod3a.analyze_v4(v2.ord)
                if verbose:
                    print(amod3a.data[1])
                    print([hex(x) for x in buf])
                t, v = amod3a.find_v4(v2.ord)
                assert t == "2B", (t, hex(v))
                ax1 = ax * Xsp2_Co1('c', v) ** -1
                ax1_types = [ax1.axis_type(e) for e in range(1,3)]
                assert "2A" in ax1_types
                ## Todo: check scalar product
                t_markers = axis_count_BpmC(ax1)
                #print(t_markers)
                assert set(t_markers) == {0, 28}
                e = t_markers.index(28) + 1
                assert ax1.axis_type(e) == '2A', ax1.axis_type(e)

    print("passed") 


#########################################################################
# Generate test cases (axis of type 2B, baby axis)
#########################################################################



def iter_2B_axis_pairs(ntests=10):
    baby_axes = []
    for _, g in AXIS_DICT.values():
        baby_axes.append(NEG_AXIS * MM(g))
    for i in range(ntests):
        for n, baby_axis in enumerate(baby_axes):
            h = Xsp2_Co1('r')
            ax = AXIS_Y * h
            baby = baby_axis * h
            yield n+1, ax, baby


#########################################################################
# Check mattrices in cases where Hadamard transform applies
#########################################################################

H256 = np.array(hadamard(256, dtype = np.int32) & 127, dtype = np.int8)


def check_analyze_hadamard(ax, baby, amod3):
    a_data = amod3.raw_data_flat[32*32:48*32]
    a_data_in = a_data[8*32:]
    a_data_transformed = a_data[:8*32]
    data = np.zeros(256, dtype = np.uint8)
    std_type2 = std_typ2_list()
    for i in range(120):
        b = TYPE2BASIS[i]
        vector = int(std_type2[i])
        va = mm_op_eval_A(15, baby.v15.data, vector) % 3
        leech2 = XLeech2(vector)
        ax_v = ax.v15[leech2] % 3
        assert ax_v != 0, (i, b, ax_v)
        vb = ax_v * baby.v15[leech2] % 3
        data[b] =  COEFF[3*vb + va]
    if not (a_data_in == data).all():
        print(data)      
        print(a_data_in) 
        raise ValueError("Loading data from baby axis failed")    
    transformed = (data @ H256) % 128
    if not (a_data_transformed == transformed).all():
        print(transformed)      
        print(a_data_transformed) 
        raise ValueError("Transforming data from baby axis failed")    


def vector_partition_from_amod3(amod3):
    assert isinstance(amod3, MMOpFastAmod3)
    slack = amod3.raw_data_flat_uint32[24*8+6:]
    basis = slack[0:64:8]
    data = amod3.raw_data_flat[32*32 : 32*32+256]
    d = defaultdict(set)
    for i, value in enumerate(data):
        b = TYPE4IBASIS[i]
        if b:
            d[int(value)].add(int(bit_mat_mul(b, basis)))
    assert sum(len(x) for x in d.values()) == 135
    return dict(d)   
 
def best_of_vector_partition(d):
    order = [(len(v), k) for k, v in d.items()]
    selected = min(order)[1]
    #print("sel:", selected)
    v_set = d[selected]
    best = min((gen_leech2_coarse_subtype(v), v) for v in v_set)
    return best[1]



       
def profile_from_vector_partition(d, case = None):
    table = sorted([(len(d[x]), x) for x in d.keys()])
    return [(x, v) for v, x in table]




def check_hadamard(case, amod3, ref_partition = None, verbose = 0):
    vector_partition = vector_partition_from_amod3(amod3)
    if ref_partition is not None:
        assert vector_partition == ref_partition
    v_best = best_of_vector_partition(vector_partition)
    return v_best

#########################################################################
# Generate test cases (axis of type 2B, baby axis)
#########################################################################




NON_HADAMARD_CASES = [2, 3, 4]
GOOD_BABY_AXES_TYPES = set([
        '2A', '2B', '4A', '4B', '4C', '6A', '6C'
])

CASE_IMAGES = {}
for case, d in target_dicts().items():
    CASE_IMAGES[case] =  set(d.keys())
for case in NON_HADAMARD_CASES:
    CASE_IMAGES[case] = set(['2A'])
for case, s in CASE_IMAGES.items():
    assert s.issubset(GOOD_BABY_AXES_TYPES)   

UNIT_8 = np.eye(8, dtype = np.uint8)
H256 = hadamard(256, dtype = np.int32) 
H256 = np.array(H256 & 127, dtype = np.int8)

def check_case_2B(ntests = 1, extra_checks = 1, verbose = 0):
    assert use_mmgroup_fast
    TEXT = "Test %d, case %s, baby axis type = %s"
    print("Test reduction of axis type 2B") 
    for n, (case, ax, baby) in enumerate(iter_2B_axis_pairs(ntests)):
        if verbose > 1:
            print(TEXT % (n+1, case, baby.axis_type()))
        amod3 = axis_to_MMOpFastMatrix(ax, baby)
        ax_type, result = amod3.find_v4()
        assert ax_type == '2B', ax_type

        if extra_checks:
            amod3a = axis_to_MMOpFastMatrix(ax, baby)
            buf = amod3a.analyze_v4()
            g = MM('a', buf[5:5+8])
            ax_reduced = ax * g**-1
            baby_reduced = baby * g**-1
            if case not in NON_HADAMARD_CASES:
                assert buf[0] == 0x32
                part = check_analyze_hadamard(
                        ax_reduced, baby_reduced, amod3a)
                assert ax_type == '2B', ax_type
                v_best = check_hadamard(case, amod3, part)
                assert result == v_best
            else:
                assert buf[0] == 0x12
                assert (amod3a.data[1] == baby_reduced['A'] % 3).all()
                assert (amod3a.data[1,:8] == 0).all()
                if case in [3,4]:
                    assert (amod3.data[1,8:16,8:16] == UNIT_8).all()
  
        g = Xsp2_Co1('c', result) ** -1
        ax1, baby1 = ax * g, baby * g

        t_markers = axis_count_BpmC(ax1)
        #print(t_markers)
        assert set(t_markers) == {0, 28}
        e = t_markers.index(28) + 1
        assert ax1.axis_type(e) == '2A', ax1.axis_type(e)
        assert baby1.axis_type(e) in CASE_IMAGES[case], baby1.axis_type(e)
    if verbose:
        print("Cases of transformed baby axis types")   
        for case, s in CASE_IMAGES.items():
            print(" ", "%2d" % case, s)
    print("passed") 




#########################################################################
# Main program
#########################################################################



def test_2B_all(n_tests = 10):
    check_case_2B(1 + n_tests // 10, extra_checks = 1, verbose = 0)
    check_case_2Bx(n_tests)


if __name__ == "__main__":
    test_2B_all()

