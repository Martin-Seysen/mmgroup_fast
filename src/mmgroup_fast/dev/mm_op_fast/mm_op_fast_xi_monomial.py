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

import numpy as np

from mmgroup.generate_c import c_snippet, TableGenerator, make_table
from mmgroup.generate_c import UserDirective, UserFormat



from mmgroup import mat24
from mmgroup.bitfunctions import v2, bw24

from mmgroup.dev.mm_basics.mm_tables import MM_OctadTable
from mmgroup.dev.mm_basics.mm_tables_xi import MM_TablesXi


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


def compress_table(table, d=2, rowsize=32):
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
    """Deprecated"""
    assert 0 <= i < 5 and 0 <= j < 2
    if i in [1, 2, 4] or (i == 3 and j == 0):
        return False, i, j
    inv_i, inv_j = invert_stage(i, j) 
    return True, inv_i, inv_j


######################################################################
# Test functions
######################################################################


def _spread8(x):
    return [[255 & -((x >> j) & 1) for j in range(8)]]


def _comment(i):
    boxes = [MM_TablesXi.TABLE_BOX_NAMES[i,j] for j in (0,1)]
    if boxes[0] == boxes[1]:
        args = boxes[0][0], boxes[0][1]
        return "// Map box %s to box %s\n" % args
    else:
        fmt =  "// Map box %s to box %s if exponent is %d\n"
        s = ""
        for exp1, (src, dest) in enumerate(boxes):
            s += fmt % (src, dest, exp1 + 1)
        return s



###########################################################################
# Make sign tables for stage 0, i.e. box 'BC'
###########################################################################

def _spread32(x):
    return [[255 & -((int(x) >> j) & 1) for j in range(32)]]



def make_sign_tables_bc():
    t = MM_TablesXi()
    sign_indices_bc = [[], []]
    sign_data_bc = []
    sign_dict = {}
    SIGN_TAB = t.SIGN_TABLES
    LEN = t.SHAPES[0][0][1]
    for i in range(2):
        SIGN_TAB = t.SIGN_TABLES[0][i]
        for j in range(LEN):
            sign = SIGN_TAB[j]
            if sign not in sign_dict:
                sign_dict[sign] = len(sign_data_bc)
                sign_data_bc.append(_spread32(sign))
            sign_indices_bc[i].append(sign_dict[sign])
    return sign_indices_bc, sign_data_bc
     


###########################################################################
# Make sign tables for other stages
###########################################################################


def make_shuffle_entry(d, pi):
    assert len(pi) << d <= 32, (d, pi)
    res = [(i << d) ^ x ^ y  for i, x in enumerate(pi) 
        for y in range(1 << d)]
    res += [i for i in range(len(res),32)]
    return [res]

def make_tables_all():
    t = MM_TablesXi()
    PERM_TAB, SIGN_TAB = t.PERM_TABLES, t.SIGN_TABLES
    SHAPES = t.SHAPES
    table = []
    data_sign = []
    data_perm = [] 
    indices = []
    dict_sign = {}
    dict_perm = {}
    _dict_compressed = {}
    done = {}
    offset = np.zeros((5,2,2), dtype = np.uint32)
    items = [((i, j) not in ACTIVE_SETS, i, j) for i in range(1,5) 
         for j in range(2)]
    items.sort()
    for invert, i, j in items:
        signs, perms = SIGN_TAB[i][j], PERM_TAB[i][j]
        #src_shapes = SHAPES[i][0]
        dest_shapes = SHAPES[i][1]
        rowsize = dest_shapes[2]
        d, res, dt = compress_table(perms, 2, rowsize)
        _dict_compressed[(i,j)] = d, res, dt
        offset[i][j][0] = len(table)
        table += list(res)
        if invert:
            i_inv, j_inv =  invert_stage(i, j)
            assert (i_inv, j_inv) in done
            assert (i_inv, j_inv) in ACTIVE_SETS
            assert j_inv == 1 - j
            assert SHAPES[i][0] == SHAPES[i_inv][1]
            assert SHAPES[i][1] == SHAPES[i_inv][0]
            assert  _dict_compressed[i, j][0] == (
                      _dict_compressed[i_inv, j_inv][0]) 
            offset[i][j][1] = offset[i_inv][j_inv][1]
        else:
            offset[i][j][1] = len(indices)
            assert len(signs) == len(dt)
            for pi, sign in zip(dt, signs):
                if (d, pi) not in dict_perm:
                    dict_perm[(d,pi)] = len(data_perm)
                    data_perm.append(make_shuffle_entry(d, pi))
                indices.append(dict_perm[(d, pi)]) 
                if  sign not in dict_sign:
                    dict_sign[sign] = len(data_sign)
                    data_sign.append(_spread32(sign))
                indices.append(dict_sign[sign])
        done[(i,j)] = 1 

    print("DATA", len(data_perm), len(data_sign))
    table = np.array(table, dtype = np.uint8)
    print("table.shape", table.shape)
    indices = np.array(indices, dtype = np.uint8)
    print("indices.shape", indices.shape)
    data_perm = np.array(data_perm, dtype = np.uint8)
    print("data_perm.shape", data_perm.shape)
    data_sign = np.array(data_sign, dtype = np.uint8)
    print("data_sign.shape", data_sign.shape)
    indices[0::2] += len(data_sign)

    return offset, table, indices, data_perm, data_sign        





class Tables:
    t = MM_TablesXi()
    sign_indices_bc, sign_data_bc = make_sign_tables_bc()
    offset, table, indices, data_perm, data_sign  = make_tables_all() 

    @classmethod
    def comment(self, i):
        boxes = [MM_TablesXi.TABLE_BOX_NAMES[i,j] for j in (0,1)]
        if boxes[0] == boxes[1]:
            args = boxes[0][0], boxes[0][1]
            return "// Map box %s to box %s\n" % args
        else:
            fmt =  "// Map box %s to box %s if exponent is %d\n"
            s = ""
            for exp1, (src, dest) in enumerate(boxes):
                s += fmt % (src, dest, exp1 + 1)
            return s
   
    tables = {
        "MM_TABLE_SHAPES_XI": t.SHAPES,
        "U64_SPREAD8_TABLE" : [_spread8(x) for x in range(256)],
        "SIGN_INDICES_BC_TABLE_XI": sign_indices_bc,
        "SIGN_DATA_BC_TABLE_XI": sign_data_bc,
        "MM_FAST_OFFSET_XI": offset,
        "MM_FAST_TABLE_XI": table,
        "MM_FAST_INDICES_XI": indices,
        "MM_FAST_DATA_PERM_XI": data_perm,
        "MM_FAST_DATA_SIGN_XI": data_sign,
    }
    directives = {
        "OP_XI_COMMENT" : UserDirective(_comment, "i")
    }


if __name__ == "__main__":
    import mmgroup
    MM_TablesXi().display_config()





