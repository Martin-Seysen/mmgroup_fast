import os
import sys
import subprocess
import numpy as np

MY_PY_PATH = os.path.abspath(os.path.join('..', '..', '..'))
_DIR = os.path.split(__file__)[0]
PY_FILENAME = os.path.join(_DIR, "order_vector_data.py")

from mmgroup import MM, MMVector
from mmgroup.dev.mm_reduce.find_order_vector import find_element_of_order

####################################################################
## Compute vector v59 and order vector s59 from group element g59
####################################################################

def str_data(text, data):
    s = "%s = [\n   " % text
    for i, x in enumerate(data):
        s += hex(x) + ","
        s += "\n   " if i % 6 == 5 else " "
    s += "\n]\n"
    return s
    
def s59_from_v59(g59, v59):
    v = v59.copy()
    s59 = v.copy()
    for j in range(58):
        v *= g59
        s59 += v
    assert v59 * g59 != v59
    assert s59 * g59 == s59
    for j in range(24):
         if s59["X", 0, j]:
             return g59, v59, s59
    raise ValueError("Bad vector pair g59, v59")

def s59_from_g59(g59, v59=None):
    r"""Compute a vector stabilized by a group element of order 59

    The function takes an element ``g59`` of order 59 of the
    monster and a vector ``v59``  in the representation of the
    monster modulo 3, such that the vector
    ``s59 = sum(v59 * g59**i for i in range(59))`` is not trivial.
    Then ``s59`` is stabilized by ``g59``. More precisely, the
    projection of ``s59`` onto the 196883-dimensional irreducible
    representation of the monster is not trivial.

    The function returns the triple ``(g59, v59, s59)``.  Here
    ``g59`` and ``s59`` are instances of class ``MM``, and ``s59``
    is a vector which is an intance of  class ``MMVector``.

    If input ``v59`` is ``None``, a random vector is generated.
    """
    if v59:
        return s59_from_v59(g59, v59)
    for i in range(24):
        try:
            v59 = MMVector(3, "X", 0, i)
            return s59_from_v59(g59, v59)
        except:
            continue
    raise ValueError("Bad vector g59")

####################################################################
## Write order vector data to file
####################################################################

HEADER = """# This file has been created automatically, do not change!
# For documentation see module
# mmgroup.dev.mm_reduce.find_order_vector.start_vector_59.py.
"""

def write_vector_59_mod3(g59, v59):
    """Write group element g59 and vector v59 to file"""
    print("Writing file " + PY_FILENAME)
    assert isinstance(g59, MM)
    assert isinstance(v59, MMVector) and v59.p == 3
    f = open(PY_FILENAME, "wt")
    print(HEADER, file = f)
    a_v59 = v59.as_sparse()
    a_g59 = g59.mmdata
    for text, data in [("G59", a_g59), ("V59", a_v59)]:
         print(str_data(text, data), file = f)
    f.close()

####################################################################
## Read order vector data from file
####################################################################

def read_vector_59_mod3(recompute = False, verbose = 0):
    r"""Compute a vector stabilized by a group element of order 59

    The function computes an element ``g59`` of order 59 of the
    monster and a vector ``v59``  in the representation of the
    monster modulo 3, such that the vector
    ``s59 = sum(v59 * g59**i for i in range(59))`` is not trivial.
    Then ``s59`` is stabilized by ``g59``. More precisely, the
    projection of ``s59`` onto the 196883-dimensional irreducible
    representation of the monster is not trivial.

    The function returns the tuple ``(g59, v59, s59)``.  Here
    ``g59`` and ``s59`` are instances of class ``MM``; and ``s59``
    is a vector which is an instance of  class ``MMVector``.

    ``g59`` and ``v59`` are read from a file if possible.
    ``recompute = True`` forces recomputation of these values.
    """
    process_args = [sys.executable, "start_vector_59.py", "-w"]
    sys.path.append(MY_PY_PATH)
    import mmgroup_fast
    try:
        assert not recompute
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59
    except:
        subprocess.check_call(process_args, cwd=_DIR)
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59
    sys.path.pop()
    g59 = MM('a', G59)
    v59 = MMVector(3, 'S', V59)
    return s59_from_g59(g59, v59)

####################################################################
## Concatenate components of vector
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
## Tables for code generation
####################################################################

class Tables:
    directives = {}
    def __init__(self):
        s59 = read_vector_59_mod3()[2]
        sABC = concatenate_vector(s59, "ABC")
        sT = concatenate_vector(s59, "T")
        sXZY = concatenate_vector(s59, "XZY")
        self.tables = {
            "ORDER_VECTOR_59_ABC": sABC,
            "ORDER_VECTOR_59_T": sT,
            "ORDER_VECTOR_59_XZY": sXZY,
        }

class MockupTables:
    directives = {}
    a = np.zeros((1,24))
    tables = {
            "ORDER_VECTOR_59_ABC": a,
            "ORDER_VECTOR_59_T": np.zeros((1,32)),
            "ORDER_VECTOR_59_XZY": a,
    }

####################################################################
####################################################################
# Main program for testing and function find_element_of_order
####################################################################
####################################################################

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # sometimes needed
    if "-w" in sys.argv:
        for i in range(10):
            try:
                s_g59 = find_element_of_order(59)
                g59, v59, s59 = s59_from_g59(MM(s_g59))
                break
            except:
                #raise
                continue
        try:
            write_vector_59_mod3(g59, v59)
        except:
            raise ValueError("No order vector found")
    if "-v" in sys.argv:
        g59, v59, s59 = read_vector_59_mod3(recompute=False, verbose=1)
        print("g59 =", g59)
        print("v59 =", v59)
