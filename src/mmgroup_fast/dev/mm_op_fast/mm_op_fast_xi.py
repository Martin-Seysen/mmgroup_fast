from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import sys
import os
import collections
import re
import warnings
from numbers import Integral
from random import randint
from functools import reduce
from operator import __or__



from mmgroup.generate_c import c_snippet, TableGenerator, make_table
from mmgroup.generate_c import UserDirective, UserFormat



from mmgroup import mat24
from mmgroup.bitfunctions import v2, bw24

from mmgroup.dev.mm_basics.mm_tables import MM_OctadTable
from mmgroup.dev.mm_basics.mm_tables_xi import MM_TablesXi
from mmgroup.dev.mm_op.mm_op import MM_Op


###########################################################################
# Compress tables for operator xi
###########################################################################



def diff_table(tbl, d_max):
    """yet to be dococumented"""
    if len(tbl) == 0:
        return 0, tbl
    for d in range(d_max):
        lt = len(tbl)
        if lt & 1:
            return d, tbl
        q = 1 << d
        f = reduce(__or__, [tbl[i] ^ tbl[i+1] ^ q for i in range(0, lt, 2)])
        if f:
            return d, tbl
        tbl = tbl[0 : lt : 2]
    return d_max, tbl


def compress_table(table, d, rowsize=32):
    """yet to be dococumented"""
    assert 0 <= d <= 2
    assert len(table) % rowsize == rowsize % (1 << d) == 0
    d, tbl = diff_table(table, d)
    q = 1 << d
    res, dt = list(zip(*[divmod(x, q) for x in tbl]))
    dd = rowsize >> d
    dt_rows = [tuple(dt[i:i+dd]) for i in range(0, len(dt), dd)]
    #print(dt32)
    #if d:
    #     assert 0 <= min(res) <= max(res) < 256
    return d, res, dt_rows


ACTIVE_SETS = [(0,0), (0,1), (1,0), (1,1), (3,0), (4,0), (4,1)] 

def analyze_tables():
    """yet to be dococumented"""
    t = MM_TablesXi()
    PERM_TAB, SIGN_TAB = t.PERM_TABLES, t.SIGN_TABLES
    SHAPES = t.SHAPES
    TOTAL_SIGN = set()
    TOTAL_DIFF = set()
    TOTAL_ALL = set()
    TOTAL_LEN = 0
    for i in range(5):
        for j in range(2):
            signs, perms = SIGN_TAB[i][j], PERM_TAB[i][j]
            L_SIGN = len(set(signs))
            rowsize = SHAPES[i][1][2]
            d, res, dt = compress_table(perms, 2, rowsize)
            assert len(signs) == len(dt), (len(signs), len(dt), rowsize)
            d_list = [d] * len(dt)
            all = list(zip(d_list, signs, dt))
            assert len(all) == len(dt)
            #if (i,j) == (2,1): print(all)
            L_ALL = len(set(all))
            assert len(res) << d == len(perms)
            L_DIFF = len(set(dt))
            if (i,j) in ACTIVE_SETS:
                TOTAL_SIGN |= set(signs)
                TOTAL_ALL |= set(all)
            if i != 2:
                TOTAL_DIFF |= set([(d, x) for x in  dt])
            TOTAL_LEN += len(res)
            print("%d, %d: %4d %4d (%4d), d=%d" % 
                (i, j, L_SIGN, L_DIFF, L_ALL, d))
    print("Total: %4d, %4d (%3d), len=%d" % 
        (len(TOTAL_SIGN), len(TOTAL_DIFF), len(TOTAL_ALL), TOTAL_LEN))

       
###########################################################################
# Create tables
###########################################################################


def invert_stage(i, j):
    assert 0 <= i < 5 and 0 <= j < 2
    if i < 2:
        return i, 1-j
    return 6 - i, 1 - j

def invert_shuffle_sign(i, j):
    assert 0 <= i < 5 and 0 <= j < 2
    if i in [1, 2, 4] or (i == 3 and j == 0):
        return False, i, j
    inv_i, inv_j = invert_stage(i, j) 
    return True, inv_i, inv_j


######################################################################
# Test functions
######################################################################





if __name__ == "__main__":
    import mmgroup
    MM_TablesXi().display_config()





