from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import sys
import os
import collections
import re
import warnings
from numbers import Integral
from array import array
from random import randint

import numpy as np

from mmgroup.generate_c import c_snippet, TableGenerator, make_table
from mmgroup.generate_c import UserDirective, UserFormat
from mmgroup import mat24
from mmgroup.bitfunctions import v2, bw24
from mmgroup.dev.mm_basics.mm_tables import MM_OctadTable
from mmgroup.dev.mm_op.mm_op import MM_Op
from mmgroup.dev.mm_op.mm_op_xy import Perm64_xy
from mmgroup.dev.mat24.mat24tables import Mat24Tables
            


###########################################################################
# Scalar products and inversions of a 2048 x 24 vector
###########################################################################


class ScalarProd2048:

    def __init__(self, *args, **kwds):
        self.tables = self.make_tables()
        self.directives =  self.make_directives()
        self.Perm64_xy = None
        

    @staticmethod
    def bit_table_from_64_bit_int(int64, start = "", end = ""):
        data = ["0F"[(int64 >> j) & 1] for j in range(64)]
        s  = " %s{" % start + ", ".join(data[:16]) + ",\n"
        s += "  " + ", ".join(data[16:32]) + ",\n"
        s += "  " + ", ".join(data[32:48]) + ",\n"
        s += "  " + ", ".join(data[48:64]) + "}" + end + "\n"
        return s

    ######################################################################
    # Generate tables for scalar products
    ######################################################################

    def scalar_prod_table_entry(self, v1):
        v = mat24.gcode_to_vect(v1)
        data = ["0F"[(v >> j) & 1] for j in range(24)]
        return "{{" + ", ".join(data) + "}}"            

    def scalar_prod_pwr2_table(self, n):
        entries = []
        for i in range(0, n + 1):
            entries.append(self.scalar_prod_table_entry((1 << i) - 1))
        return "  " + ",\n  ".join(entries)  + "\n"

    def scalar_prod_index_table(self, vmax, dist):
        data = [v2(x) + 1 for x in range(dist, vmax, dist)] + [0]
        return "  " +  ", ".join(map(str, data))  + "\n"            

    def permute_64_parity_table(self, *args):
        wt = MM_OctadTable.perm64_weights
        return self.bit_table_from_64_bit_int(wt)

    def xy_table_value(self, index):
        if self.Perm64_xy is None:
            self.Perm64_xy = Perm64_xy()
        return self.Perm64_xy.table_value(index)

    def xy_table(self, stop, step):
        ind = list(range(0, stop, step))
        values = [self.xy_table_value(x) for x in ind]
        starts = ["{"] * len(values)
        ends = ["}," for i in range(len(values) - 1)] + ["}"]
        arglist = zip(values, starts, ends)
        data = [self.bit_table_from_64_bit_int(*arg) for arg in arglist]
        return "".join(data)

    def octad_expand64_table(self):
        octad_in = Mat24Tables.octad_table
        octad_out = np.zeros((759, 1,  32), dtype = np.uint32)
        SUBOCT = [0,1,2,4, 8,16,32,63]
        for i in range(759):
            o_in = octad_in[8*i:]
            for j in range(8):
                octad_out[i, 0, o_in[j]] = SUBOCT[j] 
                octad_out[i, 0, 24+j] = o_in[j]
        return octad_out


    def make_directives(self):
        return {
            "SCALAR_PROD_2048_PWR2_TABLE": 
                   UserDirective(self.scalar_prod_pwr2_table, "i"),
            "SCALAR_PROD_2048_INDEX_TABLE": 
                   UserDirective(self.scalar_prod_index_table, "ii"),
            "PERMUTE_64_PARITY_TABLE": 
                   UserDirective(self.permute_64_parity_table, "."),
            "TABLE_PERM64_XY": 
                   UserDirective(self.xy_table, "ii"),           
        }


    @staticmethod
    def lsbit(x):
        return (x & -x).bit_length() - 1

    @staticmethod
    def bitweight6(x):
        return bw24(x & 0x3f)

    def make_tables(self):
        return {
            "lsbit": UserFormat(self.lsbit),
            "bitweight6": UserFormat(self.bitweight6),
            "OCTAD_EXPAND_TABLE": self.octad_expand64_table(),
        }






######################################################################
# Summarizing the tables given above
######################################################################

class Tables:
    def __init__(self, **kwds):
        self.tables = {}
        self.directives = {}
        table_classes = [ ScalarProd2048() ]
        for t in table_classes:
            self.tables.update(t.tables)
            self.directives.update(t.directives)


######################################################################
# Test functions
######################################################################





if __name__ == "__main__":
    c = Perm24_Standard(7)
    print(c.prepare("perm", "perm_fast"))
    for i, x in c.expressions("source", "perm_fast", "sign"):
          print("dest[%s] =\n  %s;" % (i, x))
    print("1234567890"*7)
    print("\n//\n")

    c = Perm24_Benes(7)
    print(c.prepare("net", "result"))
    print(c.declare("v"))
    print(c.load("source"))
    print(c.permute("maask", "sign"))
    print(c.store("dest"))
    1/0


    with open("tmp.c", "wt") as f: f.write(SmallPerm64(3)._test())
    os.system("gcc -c tmp.c")
    os.system("del tmp.c tmp.o")
    1/0

    for p in [3, 7, 127]:
        print ("\ntest", p)
        a = SmallPerm24(p)
        a._test()





