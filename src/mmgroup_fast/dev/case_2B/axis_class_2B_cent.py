import sys
import time
import os
from math import floor
from random import randint, shuffle, sample
from collections import defaultdict
from multiprocessing import Pool
from optparse import OptionParser

import numpy as np

sys.path.append(r"C:\Data\mmgroup\src")

from mmgroup.bitfunctions import lin_table, bit_mat_inverse, bit_mat_mul
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_reduce_type4
from mmgroup.generators import gen_ufind_init, gen_ufind_union
from mmgroup.generators import gen_ufind_find_all_min, gen_ufind_make_map 
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup import MM, AutPL, PLoop, Cocode, XLeech2, Xsp2_Co1, MMV, GcVector
from mmgroup.axes import Axis, BabyAxis



##################### marking an axis ##############################################


from axis_class_2B_sub import Y, Y_Gx0, AXIS_Y
from axis_class_2B_sub import short_E8_vectors


_AXIS_SAMPLES = None


D_T_EXPECTED = {(0, 2): 1, (2, 4): 255, (4, 4): 135, (2, 2): 120, (0, 4): 1}
D_T_EXPECTED = {(0, 2): 256, (0, 0): 136, (2, 2): 120}

def get_axis_samples():
    global _AXIS_SAMPLES
    if _AXIS_SAMPLES is not None:
        return _AXIS_SAMPLES
    T1, T2 = MM('t', 1), MM('t', 2)
    _AXIS_SAMPLES = []
    leech_basis = [XLeech2(0, Cocode([0, x])).ord for x in range(1, 7)]
    leech_basis += [0x800000, 0x1000000, XLeech2(PLoop(range(8)), 0).ord]
    data = lin_table(leech_basis)
    for d in data:
        #print("d", hex(d))
        t1 = gen_leech2_type(d)
        t2 = None
        md = MM(Xsp2_Co1(XLeech2(d)) * Y_Gx0)
        #print("md=", md)
        for T in (T1, T2):
            try:
                #print(T, md ** T)
                d1 = XLeech2(md ** T).ord
                #print(hex(d1))
                t2 = gen_leech2_type(d1)
            except:
                pass
        assert t2 is not None
        t1, t2 = t1 % 4, t2 % 4
        types = tuple(sorted([t1, t2]))
        _AXIS_SAMPLES.append((d, types))
    d_t = defaultdict(int)
    for _, types in _AXIS_SAMPLES:
        d_t[types] += 1
    assert dict(d_t) == D_T_EXPECTED
    return _AXIS_SAMPLES



def cent_axis(axis):
    d = defaultdict(int)
    for x, types in get_axis_samples():
        ax = axis * Xsp2_Co1(XLeech2(x))
        if axis == ax:
            d[types] += 1
    d_list = []
    for t in sorted(d):
        d_list.append((t, d[t]))
    return d_list
        
    


if __name__ == "__main__":
    for i in range(100000):
         get_axis_samples()    
    print(cent_axis(AXIS_Y))


         
        
