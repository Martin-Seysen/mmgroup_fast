import os
from pathlib import Path
import re
from collections import defaultdict
from multiprocessing import Pool
import numpy as np


from mmgroup import PLoop, GCode, Cocode, XLeech2, GcVector, Octad
from mmgroup import MM, AutPL, Xsp2_Co1, MMV, MMSpace
from mmgroup.axes import Axis, BabyAxis
from axis_class_2B_sub import analyse_axis_mark
from compute_order import compute_order


script_dir = Path(__file__).resolve().parent
CERTIFICATE_PATH = os.path.join(script_dir, "certificate.txt")

# Standard classes implementing the Monster, G_x0, the Leech lattice
# mod 2, and the Griess algebra.
# Elements of the Monster are used for transforming vectors
# in the Griess algebra only.
# Elements of G_x0 are used for transforming vectors in the 
# Griess algebra or vectors in the Leech lattice mod 2.
from mmgroup import MM0, MMV, Xsp2_Co1, XLeech2

G_x0 = Xsp2_Co1   # implements an element of G_x0
M = MM0           # implements an element of the Monster M
Griess = MMV(15)  # implements an element of the Griess Algebra mod 15

# The standard axis v^+ in the Griess algebra
STD_AXIS = Griess("A_2_2 - A_3_2 + A_3_3 - 2*B_3_2")
assert Axis().v15 == STD_AXIS

# Y is the standard 2A involution such that x_{-1} * y is in class 2B
# We have Y = y_o, where o is the standard octad.
Y = MM('y', PLoop(range(8)))
# Y_Gx0 is Y as an element of G_x0
Y_Gx0 = Xsp2_Co1(Y)
# AXIS_Y is the axis of involution Y
AXIS_Y = Axis('i', Y)

# CONJ_Y conjugates AXIS_Y to the standard axis v^+
CONJ_Y = Y.conjugate_involution()[1]
# CONJ_Y_INV conjugates the standard axis v^+ to AXIS_Y 
CONJ_Y_INV = CONJ_Y ** -1
assert Axis() * CONJ_Y_INV == AXIS_Y
NEG_AXIS = Axis(BabyAxis()) * CONJ_Y ** CONJ_Y_INV
NEG_AXIS.rebase()



#####################################################################
# Parsing the certificate
#####################################################################


# regular expression object for parsing a line of the certificate 
matchobj = re.compile(
    r"([a-z0-9]+\*?)\:(\s+([A-Z0-9]+))?(\s+(<[a-z0-9_*]+>))?")

# parse  a line of the certificate
def parse_line(s):
    """parse a line ``s`` of a certificate

    The function returns the content of the line ``s`` a triple
    ``(tag, value, g)``.

    ``tag``   is the tag at the beginning of the line, of type str

    ``value`` is the value after the tag; it may be an int or a str

    ``g``     is an element of the Monster given as a string

    The meaning of ``value`` and ``g`` depends on the ``tag``.
    All entries may be None if not present in the line ``s``.
    """ 
    m = matchobj.match(s)
    if m:
        tag, _, n, _, g = m.groups()
        if n is not None and n.isdigit():
            n = int(n)
        return tag, n, g
    else:
        return None, None, None


def load_axes():
    """Pass through a certificate for finding  descriptions of axes

    The function parses the lines of the file named CERTIFICATE_PATH
    and returns a dictionary ``axis_data`` containing the axes found.

    The dictionary contains keys 'fixed', 'start' such that
    ``M(axis_data['fixed'])`` maps the standard axis STD_AXIS to the
    fixed axis AXIS_Y and ``M(axis_data['start'])`` maps STD_AXIS to
    the start axis NEG_AXIS of the orbit to be considered.
    
    Dictionary ``axis_data`` also contains a key 'axes' with its value
    being another ditionary ``d``.
    That dictionary ``d`` maps the axis numbers to a pair (g, cent).
    Here ``g`` is a string such that M(g) maps the start axis
    NEG_AXIS to the current axis. ``cent`` is a list of strings
    ``s`` such that each of the elements ``G_x0(s)`` centralizes the
    current axis as well as the axis AXIS_Y.
    """
    axis_no = None
    d = {}
    axis_data = {"axes" : d}
    for s in open(CERTIFICATE_PATH):
        tag, n, g = parse_line(s)
        if tag == "axis":
            axis_no = n
            assert AXIS_Y * M(g) == AXIS_Y
            d[axis_no] = (g, []) 
            current_axis = NEG_AXIS * M(g)
        if tag == "cent" and axis_no is not None:
            assert current_axis * Xsp2_Co1(g) == current_axis
            assert AXIS_Y * Xsp2_Co1(g) == AXIS_Y
            d[axis_no][1].append(g) 
        if tag == "end":
            axis_no = current_axis = None
        if tag == "fixed":
            assert Axis() * M(g) == AXIS_Y
            axis_data[tag] = g
        if tag == "start":
            assert Axis() * M(g) == NEG_AXIS
            axis_data[tag] = g
    #print(axis_data.keys())
    #print(axis_data["axes"].keys())
    return axis_data


#####################################################################
# Watermark axes, so that they can be distinuished
#####################################################################


def watermark_axis(g):
    """Simple watermarking of axis NEG_AXIS * M(g)

    The watermaking is chosen so that all axes in the sam orbits
    obtain the same watermark.
    """
    axis = NEG_AXIS * MM(g)
    return axis.axis_type(), analyse_axis_mark(axis)

def check_watermarking_of_axes(axes):
    """Check that watermarking is different for all axes

    The certificate contains a list of axes. Here each axis is
    given by an element g of the Monster such that NEG_AXIS * M(g)
    is equal to that axis.

    The function checks that different axes have a different
    watermarking. It fails if this is not the case.
    """
    wm_dict = defaultdict(list)
    for i, (g, _) in axes.items():
        wm = watermark_axis(g)
        wm_dict[wm].append(i)
    assert len(wm_dict) == len(axes)


#####################################################################
# Compute ist of orders of cetralizers od axis orbit representatives
#####################################################################


MAX_CPUS = 20 # change to a lower value if short of memory!

# Order of the group O_8(2)
ORDER_O_8_2 = 174182400 
# Order of the centralizer H = 2^26.O_8(2) of axis AXIS_Y in G_x0.
ORDER_CENT_Gx0_YAXIS = 2**26 * ORDER_O_8_2
# Number of axes orthogonal to the axis AXIS_Y
N_FEASIBLE_AXES = 11707448673375



def compute_order_list(subgroups):
    """Return list orders of a list of subgroups of G_x0

    Here each entry of the list ``subgroups`` describes a subgroup
    of G_x0 given by a list of generators. Such a generator is
    usually a string describing an element of G_x0.

    The function returns the list of the orders of the subgroups
    presented in the list ``subgroups``. There is a negligible
    probability that an order in the returned list is too small.
    """
    n_cpu = min(MAX_CPUS, os.cpu_count())
    with Pool(processes = n_cpu) as pool:
        result = pool.map(compute_order, subgroups)
    pool.join()
    return result


 


def check_order_list(orders):
    """Check that orders of centralizer of axes ar as expected

    Let H be the centralizer of the axis AXIS_Y in G_x0. Call an
    axis *feasible* if it is orthogonal to AXIS_X. 

    The certificate contains a transversal of the H orbits on the
    feasible axes. We want to show that no element is missing in
    that transversal.

    For each axis in that transversal it also contains a generating
    set for the centralizer of the axis in H. From that generating
    set we compute (a lower bound for) the order of the centralizer
    of axis in H. That lower bound is almost certainly sharp. Since
    the order of H is known, we may also compute the size of the
    orbit of the axis under H.

    We check that the sum of these orbit sizes is equal to the
    total number of feasible axes, which is also known.
    Here the function fails in case of a mismatch.  
    """ 
    total = 0
    for o in orders:
        q, r = divmod(ORDER_CENT_Gx0_YAXIS, o)
        assert r == 0
        total += q
    assert total == N_FEASIBLE_AXES, (total, total /  N_AXES)   


#####################################################################
# Main program
#####################################################################


def check_certificate():
    print("Checking certificate...")
    axis_data = load_axes()
    axes = axis_data["axes"] 
    check_watermarking_of_axes(axes)
    cent = [c_list for (_, c_list) in axes.values()]
    orders = compute_order_list(cent)
    check_order_list(orders)
    print("Certificate check passed")


