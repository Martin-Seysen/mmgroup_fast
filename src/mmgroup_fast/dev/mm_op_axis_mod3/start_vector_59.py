import os
import sys
import numpy as np


if __name__ == "__main__":
    MY_PY_PATH = os.path.abspath(os.path.join('..', '..', '..'))
    sys.path.insert(0, MY_PY_PATH)
    import mmgroup_fast

from mmgroup import MM, MMVector
from mmgroup.dev.mm_reduce.find_order_vector import find_element_of_order



_DIR = os.path.split(__file__)[0]
PY_FILENAME = os.path.join(_DIR, "order_vector_data.py")


HEADER = """# This file has been created automatically, do not change!
# For documentation see module
# mmgroup.dev.mm_reduce.find_order_vector.start_vector_59.py.
"""


####################################################################
## Write order vector data to file
####################################################################


def str_data(text, data):
    s = "%s = [\n   " % text
    for i, x in enumerate(data):
        s += hex(x) + ","
        s += "\n   " if i % 6 == 5 else " "
    s += "\n]\n"
    return s


def write_vector_59_mod3(g59, v59):
    print("Writing file " + PY_FILENAME)
    f = open(PY_FILENAME, "wt")
    print(HEADER, file = f)
    a_v59 = v59.as_sparse()
    a_g59 = g59.mmdata
    for text, data in [("G59", a_g59), ("V59", a_v59)]: 
         print(str_data(text, data), file = f)
    f.close()
    


    


def find_vector_59_mod3(write = False, verbose = 0):
    r"""Compute a vector stabilized by a group element of order 59

    The function computes an element ``g59`` of order 59 of the
    monster and a vector ``v59``  in the representation of the 
    monster modulo 3, such that the vector
    ``s59 = sum(v59 * g59**i for i in range(59))`` is not trivial. 
    Then ``s59`` is stabilized by ``g59``. More precisely, the 
    projection of ``s59`` onto the 196883-dimensional irreducible
    representation of the monster is not trivial.
        
    The function returns the tuple ``(g59, v59, s59)``.  Here
    ``g59`` and ``s59`` are instances of class ``MM``, and ``s59``
    is a vector which is an intance of  class ``MMVector``.

    If parameter "write" is true then data are written to file.
    """
    while True:    
        s_g59 = find_element_of_order(59, verbose = verbose)
        assert isinstance(s_g59, str)
        g59 = MM(s_g59)
        for i in range(3):
            v59 = MMVector(3, "X", 0, i)
            v = v59.copy()
            s59 = v.copy()
            for j in range(58):
                v *= g59
                s59 += v
            for j in range(24):
                if s59["X", 0, j]:
                    if verbose:
                        print("g59 =", g59)
                        print("v59 =", v59, "# (mod 3)")
                    if write:
                        write_vector_59_mod3(g59, v59)
                    return g59, v59, s59


####################################################################
## Read order vector data from file
####################################################################



def read_vector_59_mod3(recompute = False, verbose = 0):
    """Read a vector stabilized by a group element of order 59

    Return value is as in function ``find_vector_59_mod3``.
    Data are read from file if possible, and recomputed if not.
    ``recompute = True`` forces recomputation.
    """
    try:
        assert not recompute
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59
    except:
        find_vector_59_mod3(True, verbose)
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59
    g59 = MM('a', G59)
    v59 = MMVector(3, 'S', V59)
    v = v59.copy()
    s59 = v.copy()
    for j in range(58):
        v *= g59
        s59 += v
    if verbose:
        print("Order vector read from file:")
        print("g59 =", g59)
        print("v59 =", v59, "# (mod 3)")
    assert s59 == s59 * g59
    assert (s59['X',0] != 0).any()
    return g59, v59, s59
   

####################################################################
## Concatenate components of vector)
####################################################################

def concatenate_vector(v, tags, verbose=0):
    vlist = [v[t] for t in tags]
    v1 = np.concatenate(vlist)
    dt = v1.dtype
    rows, cols = v1.shape
    if cols == 64:
        v1 = v1.reshape(2 * rows, cols // 2)
        rows, cols = v1.shape
    if rows % 4:
        slack = np.zeros((4 - rows % 4, cols), dtype=dt)
        v1 = np.concatenate((v1, slack))
    rows, cols = v1.shape
    new_rows = rows//4
    v2 = v1.reshape((new_rows, 4, cols))
    v3 = np.zeros((new_rows, cols), dtype=dt)
    for i in range(new_rows):
        v3[i] = sum(v2[i, j] << (6 - 2*j) for j in range(4))
    if verbose:
        print("Shape of tags %s:" % tags, v3.shape)
    return v3


####################################################################
## Concatenate components of vector)
####################################################################

class Tables:
    s59 = read_vector_59_mod3()[2]
    sABC = concatenate_vector(s59, "ABC")
    sT = concatenate_vector(s59, "T")
    sXZY = concatenate_vector(s59, "XZY")
    tables = {
        "ORDER_VECTOR_59_ABC": sABC,
        "ORDER_VECTOR_59_T": sT,
        "ORDER_VECTOR_59_XZY": sXZY,
    }
    directives = {}

####################################################################
####################################################################
# Main program for testing
####################################################################
####################################################################



def display_all():
    find_vector_59_mod3(write = True, verbose = 1)
    _, _, s59 = read_vector_59_mod3(recompute = False, verbose = 1)
    sABC = concatenate_vector(s59, "ABC")
    sT = concatenate_vector(s59, "T")
    sXYZ = concatenate_vector(s59, "XYZ")
    

if __name__ == "__main__":
    display_all()

