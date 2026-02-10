import os
from multiprocessing import Pool


from mmgroup import PLoop, GCode, Cocode, XLeech2, GcVector, Octad
from mmgroup import MM, AutPL, Xsp2_Co1, MMV, MMSpace
from mmgroup.axes import Axis, BabyAxis
from axis_class_2B_axes import rand_y
from axis_class_2B_axes import AXIS_Y, NEG_AXIS
from axis_class_2B_reduce import reduce_axis
from check_certificate import CERTIFICATE_PATH



def mmstr(g):
    return str(MM(g))[1:]

# Number of generators of the centralizer to be generated for each axis 
N_GENERATORS = 30

def str_certificate(case, g, size = N_GENERATORS):
    axis = NEG_AXIS * MM(g)
    g_0 = axis.g1
    axis.rebase()
    axis = reduce_axis(axis, case = case, check = True)
    g_0 *= axis.g1
    assert AXIS_Y * MM(g_0) == AXIS_Y
    assert AXIS_Y.product_class(axis) == '2B'
    assert NEG_AXIS * g_0 == axis
    #print("cccc", case, axis.axis_type())
    out = ["axis: %d  %s" % (case, mmstr(g_0))]
    for i in range(size):
        ax = axis.copy().rebase()
        ax *= rand_y()
        g0 = Xsp2_Co1(ax.g1)
        ax = ax.rebase()
        ax = reduce_axis(ax, case = case, check = True)
        assert ax == axis
        g1 = Xsp2_Co1(ax.g1)
        g2 = g0 * g1
        assert  axis * g2 == axis
        assert AXIS_Y * g2 == AXIS_Y
        out.append("cent: 1 %s" %  mmstr(g2))
    out.append("end:")
    return out
    
def str_fixed_axis(tag, axis):
    return "%s: 1 " % tag + mmstr(axis.g) + "\n"


POOL = True

def make_certificate(g_list, recompute = True):
    if not recompute and os.path.isfile(CERTIFICATE_PATH):
        return
    s = "Computing a certificate"
    print(s + "...")
    n_cpu = os.cpu_count()
    if POOL:
        with Pool(processes = n_cpu) as pool:
             results = pool.starmap(str_certificate, g_list)
        pool.join()
    else:
        results = []
        for args in g_list:
            results.append(str_certificate(*args))
            print("Case %d done" % args[0])
    with open(CERTIFICATE_PATH, "wt") as f:
        f.write(str_fixed_axis("fixed", AXIS_Y))
        f.write(str_fixed_axis("start", NEG_AXIS))
        for entry in results:
           for s in entry:
               f.write(s + "\n")
    print("Certificate written fo file " + CERTIFICATE_PATH)


