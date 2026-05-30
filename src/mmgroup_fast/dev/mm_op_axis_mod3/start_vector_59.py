import os
import sys
import subprocess
import numpy as np

MY_PY_PATH = os.path.abspath(os.path.join('..', '..', '..'))
_DIR = os.path.split(__file__)[0]
PY_FILENAME = os.path.join(_DIR, "order_vector_data.py")

from mmgroup import MM, MM0, MMVector
from mmgroup.dev.mm_reduce.find_order_vector import find_element_of_order
from mmgroup.dev.mm_reduce.find_order_vector import find_vector_p_mod3

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
    
def s59_from_v59(g59, v59, a59):
    r"""Compute a vector stabilized by a group element of order 59

    The function returns a vector ``s59``  in the representation of
    the  monster modulo 3, such that the vector
    ``s59 = sum(v59 * g59**i for i in range(59)) * a59`` is not
    trivial. Then ``s59`` is stabilized by ``g59 ** a59``. More
    precisely, the projection of ``s59`` onto the 196883-dimensional
    irreducible representation of the monster is not trivial.

    Here ``g59`` and ``a59`` are instances of class ``MM``
    (or ``MM0``); and  ``v59`` and the return value``s59`` are
    vectors which are instances of  class ``MMVector``.
    """
    v = v59.copy()
    s59 = v.copy()
    for j in range(58):
        v *= g59
        s59 += v
    assert v59 * g59 != v59
    assert s59 * g59 == s59
    s59 *= a59
    for i in range(10):
        for j in range(24):
            if s59["X", 0, j]:
                return s59
    raise ValueError("Bad vector tuple g59, v59, a59")


####################################################################
## Write order vector data to file
####################################################################

HEADER = """# This file has been created automatically, do not change!
# For documentation see module
# mmgroup.dev.mm_reduce.find_order_vector.start_vector_59.py.
"""

def write_vector_59_mod3(s_g, s_v, s_gA):
    """Write group element g59 and vector v59 to file"""
    a_g = MM0(s_g).mmdata
    a_v = MMVector(3,s_v).as_sparse()
    a_gA = MM0(s_gA).mmdata
    print("Writing file " + PY_FILENAME)
    f = open(PY_FILENAME, "wt")
    print(HEADER, file = f)
    for text, data in [("G59", a_g), ("V59", a_v), ("A59", a_gA)]:
        print(str_data(text, data), file = f)
    f.close()

####################################################################
## Read order vector data from file
####################################################################

def read_vector_59_mod3(recompute = False, verbose = 0):
    r"""Compute a vector stabilized by a group element of order 59


    The function returns the tuple ``(g59, v59, a59)``.  Here
    ``g59`` and ``a59`` are instances of class ``MM``; and ``s59``
    is a vector which is an instance of  class ``MMVector``.

    `These data are read from a file if possible.
    ``recompute = True`` forces recomputation of these values.
    """
    process_args = [sys.executable, "start_vector_59.py", "-w"]
    sys.path.append(MY_PY_PATH)
    import mmgroup_fast
    try:
        assert not recompute
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59, A59
    except:
        subprocess.check_call(process_args, cwd=_DIR)
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59, A59
    sys.path.pop()
    g59 = MM('a', G59)
    v59 = MMVector(3, 'S', V59)
    a59 = MM('a', A59)
    return g59, v59, a59


def compute_vector_59_mod3(g59, v59, a59):
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
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59, A59
    except:
        subprocess.check_call(process_args, cwd=_DIR)
        from mmgroup_fast.dev.mm_op_axis_mod3.order_vector_data import G59, V59, A59
    sys.path.pop()
    g59 = MM('a', G59)
    v59 = MMVector(3, 'S', V59)
    a59 = MM('a', G59)
    return s59_from_g59(g59, v59, a59)




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
        g59, v59, a59 = read_vector_59_mod3()
        s59 = s59_from_v59(g59, v59, a59)
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
                s_g, s_v, s_gA = find_vector_p_mod3(59)
                break
            except:
                #raise
                continue
        try:
            print(s_g, s_v, s_gA)
            write_vector_59_mod3(s_g, s_v, s_gA)
        except:
            raise ValueError("No order vector found")
    g59, v59, a59 = read_vector_59_mod3(recompute=False, verbose=1)
    s59 = s59_from_v59(g59, v59, a59)
    if "-v" in sys.argv:
        print("g59 =", g59)
        print("v59 =", v59)
        print("a59 =", a59)



