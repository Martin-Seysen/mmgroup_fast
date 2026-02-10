from collections import defaultdict
from random import randint, shuffle, sample, Random

import numpy as np

from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_prep_eqn
from mmgroup.clifford12 import leech2matrix_solve_eqn

from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_op import mm_aux_mmv_extract_sparse_signs
from mmgroup.mm_op import mm_aux_mmv_extract_sparse

from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup import PLoop, GCode, Cocode, XLeech2, GcVector, Octad
from mmgroup import MM, AutPL, Xsp2_Co1, MMV, MMSpace
from mmgroup.axes import Axis, BabyAxis
from mmgroup.general import Orbit_Lin2, Orbit_Elem2, Random_Subgroup
from axis_class_2B_2group import beautify_axis_abc, REF_AXES
from axis_class_2B_axes import Y, Y_Gx0, AXIS_Y
from axis_class_2B_axes import rand_y



def eqn_x(tag, i, j):
    v = MMSpace.index_to_short_mod2(tag, i, j)
    assert v > 0 
    return (v >> 12) & 0x7ff


def augment_v_data(v, data):
    a = np.array(data, dtype = np.uint32)
    mm_aux_mmv_extract_sparse(v.p, v.data, a, len(a))
    return a


D_CLASSES = {}



def display_equations(text, eqn):
    if text:
        print(text)
    al = 0
    for i, a in enumerate(eqn):
        print("%2d" % i, "".join([str((a >> i) & 1) for i in range(24)]))
        al |= a
    print("AL", "".join([str((al >> i) & 1) for i in range(24)]))


def axis_set_equ(ref_axis):
    nrows = 0
    v = ref_axis.v15.copy()
    solve_t = np.zeros(12, dtype = np.uint64)
    sp_values = []
    TAGS = "TX"
    for gcode in (range(8), range(24)):
        eqn_oct = GCode(gcode).ord
        n = leech2matrix_add_eqn(solve_t, nrows, 24, eqn_oct)
        nrows += n
    assert nrows == 2

    for tag in TAGS:
        a = v[tag]
        for i, row in enumerate(a):
            if (row == 0).all():
                continue
            for j, entry in enumerate(row):
                if entry != 0:
                    eqn = eqn_x(tag, i, j)
                    n = leech2matrix_add_eqn(solve_t, nrows, 24, eqn)
                    if n:
                        sp = MMSpace.index_to_sparse(tag, i, j)
                        sp_values.append(sp)
                        nrows += n
                    break                
    sp_values = augment_v_data(v, sp_values)
    lsp = len(sp_values)
    ok = mm_aux_mmv_extract_sparse_signs(15, v.data, sp_values, lsp) == 0
    assert ok
    equations = np.zeros(nrows, dtype = np.uint32)
    ok = leech2matrix_prep_eqn(solve_t, nrows, 24, equations) == 0
    assert ok
    #print("Ax_x", ref_axis.g_class, nrows)
    #display_equations("x equations", equations)
    return sp_values, equations

 
def solve_x_equations(axis, sp_values, equations):
    w = mm_aux_mmv_extract_sparse_signs(15, axis.v15.data, 
        sp_values, len(sp_values)) << 2
    if w < 0:
        raise ValueError("Equation has no solution_")
    x = leech2matrix_solve_eqn(equations, len(equations), w)
    return Xsp2_Co1('d', x & 0x7ff)


def solve_axis_x(axis, ref_axis, case):
    global D_CLASSES
    if case not in D_CLASSES:
        D_CLASSES[case] = axis_set_equ(ref_axis)
    sp_values, equations =  D_CLASSES[case]
    return solve_x_equations(axis, sp_values, equations)


##########################################################################

CASE_BAD_DICT_G = {}
det_rng = Random(45)
PART_T = 759
RAND_BAD = np.array(
     [det_rng.randint(1, (1 << 40) - 1) for i in range(64 * PART_T)],
     dtype = np.int64)
MAXLEN_CASE_BAD_DICT_G = {35:2, 14:2, 15:2}


def hash_T(axis):
    t = axis["T", :PART_T].ravel()
    return int(sum(t * RAND_BAD))

def case_bad_g_dict(axis, case):
    global CASE_BAD_DICT_G
    if case in CASE_BAD_DICT_G:
         return CASE_BAD_DICT_G[case]
    ref_axis = axis.copy()
    CASE_BAD_DICT_G[case] = d = {hash_T(axis) : Xsp2_Co1()}
    for i in range(200):
        g_t =  Xsp2_Co1(rand_y())
        ax = axis * g_t
        ax.rebase()
        ax, _ = beautify_axis_abc(ax, case)
        ax *= solve_axis_x(ax, ref_axis, case)
        h = hash_T(ax)
        if h not in d:
            g = (g_t * Xsp2_Co1(ax.g1)) ** -1
            assert ax * g == axis
            assert Y_Gx0 ** g == Y_Gx0
            d[h] = g
            if len(d) >= MAXLEN_CASE_BAD_DICT_G[case]:
                break
    print("Dict length for case %d is" % case, len(d))
    # print(d.keys())

    return d
    
    
def case_bad_g(axis, case):
    d = case_bad_g_dict(axis, case)
    h = hash_T(axis)
    return d[h] 

##########################################################################


KER_DICT = {}
KER_DICT_G = {0: 0, 1: 0x1000, 2: 0x1800, 3:0x800}

def min_index_ker(axis, tag):
    ai, aj = axis[tag].nonzero()
    if len(ai) == 0:
        return tag, None, None, None
    return tag, int(ai[0]), int(aj[0]), int(axis[tag, ai[0], aj[0]])




def ker_g(axis, ref_axis, case):
    global KER_DICT
    if case not in KER_DICT:
        KER_DICT[case] = [min_index_ker(ref_axis, tag) for tag in "YX"]
    result = [(0 if i is None else int(axis[tag, i, j] != value))
        for tag, i, j, value in KER_DICT[case]]
    #print(result)
    return Xsp2_Co1('x', KER_DICT_G[2*result[1] + result[0]])
    

##########################################################################


BAD_CASES = [14, 15, 35]

BAD_CASES_FOUND = defaultdict(list)
    
def reduce_axis(axis, case = None, check = True, verbose = False):
    axis, case = beautify_axis_abc(axis, case)
    assert AXIS_Y * axis.g1 == AXIS_Y
    ref_axis = REF_AXES[case]
    axis *= solve_axis_x(axis, ref_axis, case)
    if case in BAD_CASES:
         axis *= case_bad_g(axis, case)
         axis *= solve_axis_x(axis, ref_axis, case)
    axis *= ker_g(axis, ref_axis, case)
    assert AXIS_Y * axis.g1 == AXIS_Y

    if check:
        if axis != ref_axis:
            raise ValueError("Axis reduction error 1!")
        if AXIS_Y * axis.g1 != AXIS_Y:
            raise ValueError("Axis reduction error 2!")
        pass
    """
    for tag in "ABCTXZY":
        if not (axis[tag] == ref_axis[tag]).all():
             print("WTF", case, tag)
             #print(axis[tag] - ref_axis[tag])
             global BAD_CASES_FOUND
             bad = BAD_CASES_FOUND[(case, tag)]
             found = False
             for i, b in enumerate(bad):
                 if (b == axis[tag]).all():
                     found = True
                     print("Old bad case", case, tag, i)
                     break
             if not found:
                 bad.append(axis[tag])
                 print("New bad case", case, tag, len(bad))
    """
    return axis

       



    