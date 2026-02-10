import sys
import os
from pathlib import Path
import subprocess
from collections import defaultdict

import numpy as np

from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import py_prep_fixed_leech2_set as prep


##################### Correct substraction defect for Hadamard matrix  ############


def Hadamard_defect():
    v = np.zeros(256, dtype = np.int8)
    for stage in range(8):
        mask = 1 << stage
        v_in = np.copy(v)
        for i in range(256):
            if i & mask:
                v[i] = v_in[i ^ mask] - v_in[i] - 1
            else:
                v[i] = v_in[i] + v_in[i ^ mask]
    return -v & 0x7f
 

##################### marking an axis ##############################################




class Tables:
    from mmgroup_fast import dev
    #from mmgroup.generate_c import c_snippet, TableGenerator, make_table
    #from mmgroup.generate_c import UserDirective, UserFormat
    def __init__(self, *args):  
        try:
            from mmgroup_fast.dev.case_2B import py_process_axis_2B
        except:
            script_dir = Path(__file__).resolve().parent
            path = os.path.join(script_dir, "axis_class_2B.py")
            exec = [sys.executable, str(path), "--gen-code"]
            print("Executing", " ".join(exec))
            subprocess.check_call(exec)
            from mmgroup_fast.dev.case_2B import py_process_axis_2B

        from mmgroup_fast.dev.case_2B.py_process_axis_2B import BASIS_E8, COEFF
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE2BASIS
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE4IBASIS
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE4_SCAL_0_1
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import TYPE4_SCAL_8_9   
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import TABLE_AUX_E8
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import E8_SUBSPACE_8_9
        from mmgroup_fast.dev.case_2B.py_process_axis_2B import E8_SUBSET_TRIO
        COEFF_C = [COEFF[x // 4 % 3 * 3 + x % 4 % 3]  for x in range(16)]
        type4_index, type4_coord = [],[]
        for ind, coord in enumerate(TYPE4IBASIS):
            if coord:
                type4_index.append(ind)
                type4_coord.append(coord)
        self.tables = {
            "MM_TABLE_CASE2B_BASIS_E8": BASIS_E8,
            "MM_TABLE_CASE2B_BASIS_COEFF": COEFF_C,
            "MM_TABLE_CASE2B_BASIS_TYPE2BASIS": TYPE2BASIS,
            "MM_TABLE_CASE2B_BASIS_TYPE4_INDEX": type4_index,
            "MM_TABLE_CASE2B_BASIS_TYPE4_COORD": type4_coord,
            "MM_TABLE_CASE2B_BASIS_TYPE4_SCAL_0_1": TYPE4_SCAL_0_1,
            "MM_TABLE_CASE2B_BASIS_TYPE4_SCAL_8_9": TYPE4_SCAL_8_9,
            "MM_TABLE_CASE2B_BASIS_TABLE_AUX_E8": TABLE_AUX_E8,
            "MM_TABLE_CASE2B_HADAMARD_DEFECT": Hadamard_defect(),
            "MM_TABLE_CASE2B_E8_SUBSPACE_8_9": prep(E8_SUBSPACE_8_9),
            "MM_TABLE_CASE2B_E8_SUBSET_TRIO": E8_SUBSET_TRIO,
        }
        self.directives = {
        }

class MockupTables:
    tables = defaultdict(lambda x: [0])
    tables["MM_TABLE_CASE2B_BASIS_TABLE_AUX_E8"] = [[0]]
    directives = { }
    def __init__(self, *args):  
        pass

         
        
