import os

import mmgroup.mat24


if os.name == 'nt':
    mmgroup_dir = os.path.split(mmgroup.__file__)[0]
    #os.add_dll_directory(mmgroup_dir)

try:
    from mmgroup_fast.mm_op_fast import MMOpFastMatrix
    from mmgroup_fast.mm_op_fast import MMOpFastAmod3
except:
    pass