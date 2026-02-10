import sys
from pathlib import Path
import time
import os
import glob
from math import floor
from random import randint, shuffle, sample
from collections import defaultdict
from multiprocessing import Pool
import shelve
import shutil
import argparse


import numpy as np


from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_reduce_type4
from mmgroup.generators import gen_ufind_init, gen_ufind_union
from mmgroup.generators import gen_ufind_find_all_min, gen_ufind_make_map 
from mmgroup.mm_op import mm_op_eval_A, mm_aux_get_mmv_leech2
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.mm_reduce import mm_reduce_op_2A_axis_type
from mmgroup import MM, AutPL, PLoop, Cocode, XLeech2, Xsp2_Co1, MMV, GcVector
from mmgroup.axes import Axis, BabyAxis
from mmgroup.tests.axes.get_sample_axes import next_generation_pool

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from axis_class_2B_axes import Y , Y_Gx0, AXIS_Y
from axis_class_2B_axes import CONJ_Y_INV, NEG_AXIS
from axis_class_2B_axes import rand_y, rand_ty, rand_out

from axis_class_2B_sub import axis_mark_ab, analyse_axis_mark
from axis_class_2B_sub import partition_type4, AA_SHORT
from axis_class_2B_sub import small_type4, partition_type4_nonzero
from axis_class_2B_sub import partition_type4_large
from axis_class_2B_sub import find_mapper, test_find_mapper
from axis_class_2B_cent import cent_axis
from axis_class_2B_beautify import beautify_axis_octad
from axis_class_2B_beautify import test_beautify_axis_octad
from axis_class_2B_beautify import display_non_disambiguated_cases
from axis_class_2B_beautify import G_STD_OCTAD
from axis_class_2B_2group import beautify_axis_abc
from axis_class_2B_reduce import reduce_axis
from make_certificate import make_certificate
from check_certificate import check_certificate
from gen_code import generate_code



##################### marking an axis ##############################################

def axis_mark(g):
    if not isinstance(g, Axis):
        axis = NEG_AXIS * MM(g)
    else:
        axis = g
    ax_type = axis.axis_type()
    A = np.array(axis['A', :8, :8], dtype = np.uint32)
    tr1 = int((np.sum(np.diagonal(A))) % 15)
    A_sq = (A @ A) % 15
    tr2 = int((np.sum(np.diagonal(A_sq))) % 15)
    tr3 = int((np.sum(np.diagonal(A @ A_sq))) % 15)
    return ax_type, tr1, tr2, tr3, analyse_axis_mark(axis), axis_mark_ab(axis)



##################### finding axes ##############################################


#print(Axis(BabyAxis()).g)
#1/0

START_SAMPLE = "1"  # str(NEG_AXIS.g)

def spread(g):
    #g =  MM('r', 'B') ** CONJ_Y_INV 
    g =  MM(g) * rand_y() * rand_ty() * rand_out()
    assert AXIS_Y * g == AXIS_Y 
    return str(g)


def score(axis):
    return 64 - np.count_nonzero(Axis(MM(axis))['A', :8, :8])


N_SPREAD = 80
N_KEEP = 40
PROCESSES = os.cpu_count()
STAGES = 4

def explore_axes(verbose = 0):
    #global all_samples
    gv_list = [START_SAMPLE]
    sample_list =  []
    marks = set(axis_mark(START_SAMPLE))
    if verbose:
        #print("Start vector =", gv0.v)
        #_show_sample(0, sample_list[0])
        pass
    t_start = time.time()
    for i in range(STAGES):
        FACTOR = 2 if i < 2 else 1
        gv_list,  new_samples = next_generation_pool(
           gv_list,  
           marks,
           f_spread = spread if i else rand_y, 
           f_mark =  axis_mark, 
           f_score = score, 
           n_spread = N_SPREAD * FACTOR, 
           n_keep = N_KEEP * FACTOR, 
           processes = PROCESSES,
           verbose = verbose,
        )
        #print(new_samples) 
        for m in sorted(new_samples):
            if not m in marks: 
                sample_list.append((i, new_samples[m], m))
                if verbose:
                    #_show_sample(len(sample_list)-1, sample_list[-1])
                    pass
        marks |= new_samples.keys()
    t = time.time() - t_start
    #num_samples = sum(len(x) for x in all_samples.values())
    #print("Number of axes considered:", num_samples)
    #assert len(sample_list) == NUM_AXIS_CLASSES
    #mark_list = [mark[0] for stage, axis, mark in sample_list]
    #assert len(set(mark_list)) == NUM_AXIS_CLASSES
    print(len(sample_list), "samples found in %.1f seconds" % t)
    #if verbose:
    #    for s in sample_list:
    #        print(s[0], s[2])
    print("Exploring done")
    return sample_list



################# Eigenvalues ###########################################

def str_diag(diag, power = 1):
    EPS = 1.0e-8
    data = defaultdict(int)
    non_ints = []
    for x0 in diag:
        x = x0**power 
        x, im = x.real, x.imag
        assert im < EPS
        if abs(x - round(x)) < EPS:
            data[int(round(x))] += 1
        else: 
            done = False                       
            for d in non_ints:
                if abs(d - x) < EPS:
                    data[d] += 1
                    done = True
            if not done:
                data[x] = 1
                non_ints.append(x)
    strings = []
    for x in sorted(data):
        s = str(x) if type(x) == int else "%.2f" % x
        if data[x] > 1 : s += "^" + str(data[x])
        strings.append(s)
    return ", ".join(strings)




def display_eigenvals_axis(g):
    from mmgroup import MMVectorCRT
    from mmgroup.axes import Axis, BabyAxis
    #print(g)
    #ax = NEG_AXIS * MM(g)
    ax = Axis(g)
    axA = (ax.in_space(MMVectorCRT, 12) << 8)['A']
    eigen8 = str_diag(np.linalg.eigvals(axA[:8, :8]))
    eigen16 = str_diag(np.linalg.eigvals(axA[8:, 8:]))
    print("  Octad eigenvalues:", eigen8, "; others:", eigen16)


################# End of Eigenvalues #####################################




def write_axis_dict(d):
    """Write dictionary ``d`` of axes found to the file axis_dict.py"""
    HDR = """# This file has been created automatically, do not change!
#
# It contains a dictionary mapping a set of watermarks to a set of axes.
# Each axis is given by string g, such that the Axis is Axis(MM(g)).
# The values of the dictionary are pairs (stage, g), where ``stage`` 
# indicates the stage where the axis has been found.

from collections import OrderedDict

AXIS_DICT = OrderedDict({
"""
    m_list = []
    for ax_t, (stage, g)  in d.items():
        m_list.append((stage, int(ax_t[0][:-1]), ax_t[0][-1:], ax_t[:-1], g, ax_t))
    # Sort axes first
    axes_list = []
    for stage, _1, _2, _3, g, ax_t in sorted(m_list):
        axes_list.append((ax_t, (stage, g)))

    path = os.path.join(os.path.split(__file__)[0], "axis_dict.py")
    f = open(path, "wt")
    print(HDR, file = f)
    for key, value in axes_list:
        print("%s :\n    %s," % (str(key), str(value)),  file = f)
    print("})", file = f)
    f.close()
 

def compute_axes(recompute = False):
    """Compute a dictionary of sample axes:

    Maps the watermarks of the axes found to a pair (stage, g),
    so that Axis(MM(g)) is the sample axis
    """ 
    if not recompute:
        try:
            from axis_dict import AXIS_DICT
            return AXIS_DICT
        except:
            pass
           
    axes_dict = explore_axes(verbose = 1)
    d = dict()
    for stage, g1, ax_type in axes_dict:
        d[ax_type] = stage, str(g1)
    print(len(d), "axes found")
    write_axis_dict(d)
    from axis_dict import AXIS_DICT
    return AXIS_DICT


################# Find coefficients for computing partition ##############

MASK = 0x7f

# Here is a recomupted result of function find_coefficients(d)
GOOD_COEFFICIENTS = [93, 91, 39, 46, 10, 53, 41, 45, 32]
GOOD_MASK = MASK

#COEFFICIENTS = "COEFFICIENTS"


def get_minlen(lst):
        if len(lst) < 2:
            return len(lst)
        for i in range(1, len(lst)):
            if lst[i] != lst[i-1]:
                return i
        return len(lst) 




def find_coefficients_task(d, n):
    for i in range(n):
        aa = [randint(1, MASK)  for i in range(9)]
        found = True
        for _, g in d.values():
            ax = NEG_AXIS * MM(g)
            data = partition_type4(ax)
            min_len = get_minlen(data)
            sub_data = partition_type4_nonzero(ax, aa, MASK)
            if data[:min_len] != sub_data[:min_len]:
                found = False
                break
        if found:
            return aa
    return None    

def find_coefficients_multi_task(d):
    while(True):
        n_cpu = os.cpu_count()
        test_values = [(d, 100)] * n_cpu
        with Pool(processes = n_cpu) as pool:
             results = pool.starmap(find_coefficients_task, test_values)
        pool.join()
        for aa in results:
            if aa:
                 print("")
                 return aa, MASK
        print(".", end = "", flush = True)


def find_coefficients(d):
    print("Searching for coefficients")
    aa, MASK = find_coefficients_multi_task(d)
    neg = False
    n = 0
    for _, g in d.values():
        # check if list aa is to be negated
        ax = NEG_AXIS * MM(g)
        data = partition_type4(ax)
        if len(data) < 2 or not data[0] == data[1] == 1:
             continue
        n += 1 
        # If partition list starts with 1,1,... then possibly
        # negate data so that better candidate appears first
        types_list = small_type4(ax, aa, MASK)
        keys = [list(x.keys())[0] for x in  types_list]
        orders = [int(x[:-1]) for x in keys]
        neg = orders[0] > orders[1]
    assert n == 1
    if neg: 
        aa = [MASK + 1 - x  for x in aa]
    print("Coefficients found:", aa, MASK)
    return aa, MASK

def check_coefficients(d, aa, mask):
    for x in aa:
        assert x & mask == x, (hex(x), hex(mask))
    for _, g in d.values():
        ax = NEG_AXIS * MM(g)
        data = partition_type4(ax)
        min_len = get_minlen(data)
        sub_data = partition_type4_nonzero(ax * rand_y(),
                        GOOD_COEFFICIENTS, GOOD_MASK)
        assert data[:min_len] == sub_data[:min_len]

def load_coefficients(d, recompute = False):
    if recompute:
        aa, mask = find_coefficients(d)
    else:
        aa, mask = GOOD_COEFFICIENTS, GOOD_MASK
    check_coefficients(d, aa, mask)
    return aa, mask


################# Merge G_x0 orbits of axes #################################



def merge_axis(i, ax, marks, n_trials): 
    result = []
    for j in range(n_trials):
        k = marks[axis_mark(ax * rand_ty())]
        result.append([i, k])
    return result

def merge(axes_list, n_trials = 10):
    marks = {}
    axes = []
    for i, (stage, ax_t, g) in enumerate(axes_list):
         axes.append(NEG_AXIS * MM(g)) 
         marks[axis_mark(axes[i])] = i
         # print(i, stage, axis_mark(axes[i]))
    # for k, v in marks.items(): print(k, v)
    tl = len(axes_list)
    table = np.zeros(tl, dtype = np.uint32)
    assert gen_ufind_init(table, tl) == 0
    g_list = [(i, ax, marks, n_trials) for i, ax in enumerate(axes)] 

    n_cpus = max(1, min(32, os.cpu_count()))
    with Pool(processes = n_cpus) as pool:
         results = pool.starmap(merge_axis, g_list)
    pool.join()
    for l1 in results:
        for i, k in l1:
            gen_ufind_union(table, tl, i, k)
    n_part = gen_ufind_find_all_min(table, tl)
    assert n_part > 0
    uf_map = np.zeros(tl, dtype = np.uint32)
    assert gen_ufind_make_map(table, tl, uf_map) == 0           
    d = defaultdict(list)
    #print(uf_map)
    for i, rep in enumerate(uf_map):
        d[rep].append(i)
    res = []
    for i, rep in enumerate(uf_map):
        if rep == i:
            res.append([j + 1 for j in  d[i]])
    #print(d)
    return res



################# List of possible target axes ##############################

def target_dicts():
    """Return output of main function for option '-c'

    The function returns a dictionary that maps the case numbers to
    target dictionaries. Each target dictionary maps a the possible
    target axes occuring after 'reducing' that case to the
    relative numbers of their frequencies.   
    """
    d = compute_axes(False)
    d_out = {}
    coeff = load_coefficients(d,recompute = False)
    aa, mask = coeff
    for i, (_ , (stage, g0)) in enumerate(d.items()):
        ax = NEG_AXIS * MM(g0)
        case = i+1
        df = small_type4(ax, aa, mask, True)
        d_out[case] = df[0]
    return d_out



################# End of Eigenvalues #####################################

def display_axes(d, options, coeff):
    dict_ab = defaultdict(int)
    data_list = {}
    axes_list = []
    reduction_failures = 0
    for i, (ax_t_all, (stage, g0)) in enumerate(d.items()):
        ax_t = ax_t_all[:-1]
        axes_list.append((stage, ax_t, g0))
        minlen_large = None
        ax = NEG_AXIS * MM(g0)
        case = i+1
        g = str(ax.g)
        print("%2d: st %d, %3s,  %s" % 
            (case, stage, ax_t[0], dict(ax_t[-1])))
        if options.case and case not in options.case:
            continue
        if options.eigen:
            display_eigenvals_axis(g)
        if options.ab:
            data = axis_mark_ab(ax)
            check_data = axis_mark_ab(ax * rand_y())
            assert data == check_data, (data, check_data)
            print(" ", data)
            dict_ab[data] += 1
        if options.part:
            data = partition_type4(ax)
            check_data = partition_type4(ax * rand_y())
            assert data == check_data, (data, check_data)
            check_data1 = partition_type4(ax * rand_y(), aa = AA_SHORT)
            assert data == check_data1
            print("  partition of type-4 vectors", data)
            data_list[g] = data
        if options.part_large:
            data = partition_type4_large(ax)
            check_data = partition_type4_large(ax * rand_y())
            assert data == check_data, (data, check_data)
            print("  partition of type-2 vectors", data[0])
            print("  partition of type-4 vectors", data[1])
            minlen_large = data[1][0]
        if options.coeff:
            aa, mask = coeff
            df = small_type4(ax, aa, mask, True)
            print("  partition of small type-4 vectors", df)
            check_df = small_type4(ax * rand_y(), aa, mask, True)
            if len(df) == 1:
                assert check_df == check_df
            else:
                assert len(df) == len(check_df) == 2
                assert check_df[0] in df
                assert check_df[1] in df
            if minlen_large:
                assert min([min(x.values()) for x in df]) == minlen_large
        if options.cent:
            cent = cent_axis(ax)
            s = [f"{t}:{n}" for t, n in cent]
            print("  centralizer info: [%s]" % ", ".join(s))
            assert cent == cent_axis(ax * rand_y())
        if options.beautify:
            ax1 = ax * rand_y()
            beautify_axis_octad(ax1, check = 1, case = case, verbose = 1)
        if options.beautify_g2:
            vb = options.verbose
            for j in range(max(4, options.ntests)):
                ax1 = ax * rand_y() if j else ax.copy()
                ax1, _ = beautify_axis_abc(
                    ax1, case = case, check = 1, verbose = vb)
        if options.reduce:
            vb = options.verbose
            for j in range(max(4, options.ntests)):
                ax1 = ax * rand_y() if j else ax.copy()
                ax1 = reduce_axis(
                    ax1, case = case, check = 1, verbose = vb)
        if options.test_reduce:
            n_classes =  test_find_mapper(ax, options.test_reduce)
            if n_classes > 1:
                print(f"{n_classes} different classes found after reduction")
                reduction_failures += 1
    print(len(d), "axis classes found") 
    if len(dict_ab):
        print(len(dict_ab), "markers mod 3 found")
        """
        print("Ambiguous markers mod 3")
        for value, n in dict_ab.items():
            if n > 1:
                print(n, value)
        """
    if options.find_coeff or options.coeff:
        print("\nCoefficients for partition:", coeff)
    if options.test_reduce:
        if reduction_failures > 0:
            print(f"{reduction_failures} reduction tests failed")
        else:
            print("All reduction tests passed")
    if options.merge:
        print("Merging of axis classes under maximal subgroup of 2B")
        merge_list = merge(axes_list, n_trials = 10)
        for i, m in enumerate(merge_list):
           print("  %2d:" % (i+1), m)
    if options.test_octad:
        print("Test beautifying octads")
        axes = [(str(g), 80, j+1, options.verbose) 
                 for j, (_, _1, g) in enumerate(axes_list)]
        rounds = (options.ntests + 1) // 2
        for i in range(rounds):
            with Pool() as pool:
                results = pool.starmap(test_beautify_axis_octad, axes * 2)
            pool.join()
            assert results == ['ok'] * len(results)
            print(".", end = "", flush = True)
        print("\nTest of beautifying octads passed")
    if options.make_cert or options.check_cert:
        g_list = [(i+1, x[2]) for i, x in enumerate(axes_list)]
        make_certificate(g_list, options.make_cert)
    if options.check_cert:
        check_certificate()
    if options.beautify:
        display_non_disambiguated_cases()
    if options.gen_code:
        print("Generating code for reducing axes ...")
        fname = generate_code(*coeff)
        print("Generated code written to file", fname)
        from  py_process_axis_2B import test_axis    
        for _1, _2, g0 in axes_list:     
            axis = NEG_AXIS * MM(g0)
            for i in range(options.ntests):
                test_axis(axis * rand_y())
        print("Test of generated code passed")
       





             
########### remove a shelve ###############################################






 

def main():
    description = "Compute types of orthogonal pairs of 2A axes, where one axis is of type 2B\n"
    usage = "usage: python %prog [options]"
    parser = argparse.ArgumentParser(description = description,
           prog = 'sample_axis_class_2B')
    parser.set_defaults(processes=0)
    parser.add_argument("-r",  dest="recompute", action="store_true",
        help="recompute sample axes, takes a long time!")
    parser.add_argument("-e",  dest="eigen", action="store_true",
        help="compute eigenvalues of A part of axes")
    parser.add_argument("-a",  dest="ab", action="store_true",
        help="watermark from available information mod 3")
    parser.add_argument("-p",  dest="part", action="store_true",
        help="experimental partition of type-4 vectors")
    parser.add_argument("-l",  dest="part_large", action="store_true",
        help="larger experimental partition of all vectors")
    parser.add_argument("-f",  dest="find_coeff", action="store_true",
        help="find coefficients for partition of type-4 vectors")
    parser.add_argument("-c",  dest="coeff", action="store_true",
        help="use coefficients for partition of type-4 vectors")
    parser.add_argument("-m",  dest="merge", action="store_true",
        help="display axis classes merged in maximal subgroup of 2.B")
    parser.add_argument("-z",  dest="cent", action="store_true",
        help="display experimental centralizer information")
    parser.add_argument("-b",  dest="beautify", action="store_true",
        help="try to beautify the axes found")
    parser.add_argument("-g",  dest="beautify_g2", action="store_true",
        help="try to beautify the axes found, including operation in 2-subgroup"
        " (preliminary option)")
    parser.add_argument("-d",  dest="reduce", action="store_true",
        help="reduce axes to representatives of thair G_x0 orbits"
        " (preliminary option)")
    parser.add_argument("--test-octad",  dest="test_octad", action="store_true",
        help="test the beautifying of the standard octad")
    parser.add_argument("--make-cert",  dest="make_cert", action="store_true",
        help="Create a certificate for testing the axis orbits")
    parser.add_argument("--check-cert",  dest="check_cert", action="store_true",
        help="Check the certificate for testing the axis orbits")
    parser.add_argument("--gen-code",  dest="gen_code", action="store_true",
        help="Generate python code for dealing with an axis of class 2B")
    parser.add_argument("--ntests",  type = int, default = 1,
        metavar = 'N',
        help = "perform N tests (used in some tests)")
    parser.add_argument("-v",  dest="verbose", action="store_true",
        help="verbose operation" )
    parser.add_argument('--test-reduce',
        type=int,  default = 0, metavar = 'N',
        help = 'Perform N reduction tests from G_x0 to N_x0.'
        )
    parser.add_argument("--case", action="extend", nargs="+", type=int,
        metavar = 'N',
        help = "diplay information for cases N only"
        )


    options = parser.parse_args()
    d = compute_axes(options.recompute)
    coeff = load_coefficients(d, options.find_coeff)
    display_axes(d, options, coeff)



if __name__ == "__main__":
    main()
    pass


