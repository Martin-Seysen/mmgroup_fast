import sys
import os
import re
from collections import defaultdict, OrderedDict
from random import randint
import numpy as np

from mmgroup import MM, XLeech2, mat24, GcVector, Xsp2_Co1
from mmgroup.generators import gen_leech2_subtype, gen_leech2_type
from mmgroup.generators import  gen_leech3_neg
from mmgroup.generators import gen_leech3_op_vector_word
from mmgroup.generators import gen_leech3_reduce, gen_leech3to2
from mmgroup.generators import gen_leech3_add, gen_leech2to3_abs
from mmgroup.generators import gen_leech3_reduce_leech_mod3
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_echelon_eqn
from mmgroup.clifford12 import leech2_matrix_orthogonal
from mmgroup.clifford12 import leech2_matrix_radical
from mmgroup.clifford12 import leech2_matrix_expand
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis
from mmgroup.axes import Axis

from mmgroup.generate_c import c_snippet, TableGenerator, make_table
from mmgroup.generate_c import UserDirective, UserFormat



if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))


from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import parse_mat24_orbits
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import py_prep_fixed_leech2_set
from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import axis_count_BpmC

try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import transform_fixed_leech2_set         
    use_mmgroup_fast = True
except:
    use_mmgroup_fast = False




#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################


class Axis10A:
    r"""Collect data for an axis of type 10A

    The constructor takes an axis of type 'Axis' and compute the following
    members:

    g:      Element of the Monster of class MM, such that Axis(self.g)
            is the axis given in the constructor.
    
    v3:     Sorted list of dedicated vectors of type 2, 3, 3,
            and 7 corresponding to the axis and its automorphism
            group HS:2 in G_x0, with co-ordinates taken mod 3.
            Vectors are given as integers in Leech lattice mod 3
            encoding, sorted by their type in the Leech lattice.
            Vectors of type 3 are sorted by their subtype.
            For details we refer to [CS99], Ch. 10.3.5. 

    v2:     Vectors corresponding to the list v3, with co-ordinates
            taken mod 2, given as integers in Leech lattice encoding.
            The last entry (vector of type 7) is zero. 

    vtypes: List of subtypes of vectors in list v2, given as strings.
            The first three subtypes are as in method xsubtype of
            class XLeech2, given as a string of 2 digits. The last
            string is the type of the support of that vector mod 3,
            as given by method vtype of class GcVector.

    int_vtypes:  List of integers, equivalent to list vtypes.

    v2list: List of the 100 type-2 vectors v, such that v + w is of
            type 2 in the Leech lattice mod 2, for the first three 
            entries of list v2. See [CS99], Ch. 10.3.5.   

    v24list: List of the 100 type-4 vectors v, such that v + v2[0] is
            of is in the list v2list.  

    v4list: List of the 3850 vectors v2[0] + v + v' of type 4, where
            v and v' run through the list v2list. Our goal is to
            obtain one of these vectors in a reproducible way.

    d2:     Ordered dictionary mapping subtypes in Leech lattice mod 2
            to the number of vectors of that subtvpe in list v2list.
            The ordered list of sybtypes is given by self.T2.   

    d24:    Ordered dictionary mapping subtypes in Leech lattice mod 2
            to the number of vectors of that subtvpe in list v24list.
            The ordered list of sybtypes is given by self.T2.   

    d4:     Ordered dictionary mapping subtypes in Leech lattice mod 2
            to the number of vectors of that subtvpe in list v2list.
            The ordered list of sybtypes is given by self.T4. 

    Class members:

    T2:     list of keys of ordered dictionary self.d2

    T4:     list of keys of ordered dictionary self.d4

    The vector self.v3[3] in the Leech lattice mod 3 describes the
    24 times 24 matrix of 10A axis in part 300x of the Griess algebra
    uniquely. Members v3, v2, and v2list depend on that vector only.
    """
    T2 = [0x20, 0x22, 0x21]
    T4 = [0x48, 0x40, 0x42, 0x44, 0x46, 0x43]
    _v4list = _d2 = _d4 = None
    _MAP_GCODE = ["S%d", "U%d", "T%d", "S%d+", "U%d-"]

    def __init__(self, axis):
        assert isinstance(axis, Axis)
        assert axis.axis_type() == "10A"
        self.g = axis.g.copy()
        self._v2, self.v2list = self._compute_v2list(axis)
        self._find_mod3(axis)
        assert self.v2[0] == self._v2, (self.v2[0], self._v2) 

    @staticmethod
    def _compute_v2list(axis):
        buf = np.zeros(900, dtype = np.uint32)
        assert mm_reduce_analyze_2A_axis(axis.v15.data, buf) == 0
        assert buf[0] == 0xA1, buf[0]
        assert buf[3] == 101, (buf[3])
        assert gen_leech2_type(buf[4]) == 2
        v2 = buf[4]
        v2list = [v ^ v2 for v in buf[5:105]]     
        assert len(v2list) == len(set(v2list))  == 100
        return v2, v2list


    @classmethod
    def map_gcode_type(cls, i):
        return cls._MAP_GCODE[i >> 5] % (i & 31)

    def _find_mod3(self, axis, check = True):
        def v3key(v3):
            v2 = gen_leech3to2(v3)
            if v2 >> 28:
                supp = (v3 ^ (v3 >> 24)) & 0xffffff
                t =  GcVector(supp).vtype()
                h =  GcVector(supp).vtype(1)
                return "ZZ", t, 0, v3, h
            else:
                h = gen_leech2_subtype(v2)
                t = "%02x" % h
                return t, t, v2 & 0xffffff, v3, h

        v2, v2list = self._compute_v2list(axis)
        v2l0 = v2list[0]
        v2_array = np.array([x ^ v2l0 for x in v2list], dtype = np.uint32)
        basis = np.zeros(24, dtype = np.uint64)
        ln = leech2_matrix_radical(v2_array, len(v2_array), basis, 24)
        basis = basis[:ln]
        assert ln == 2, (ln, hex(v2), hex(basis[0]))
        v2l = np.zeros(4, dtype = np.uint32) 
        leech2_matrix_expand(basis, 2, v2l)
        v3l = [gen_leech2to3_abs(x) for x in v2l]
        assert v3l[0] == 0
        vtmp = v3l[2]
        for j in range(2):
            vtmp = gen_leech3_add(vtmp, v3l[1])
            if mat24.bw24((vtmp ^ (vtmp >> 24)) & 0xffffff) % 3 == 1:
                v3l[0] = vtmp
                break
        assert v3l[0] != 0
        v_keyed = [v3key(v3) for v3 in v3l]
        v_keyed.sort() 
        self.v2 = [x[2] for x in v_keyed]
        self.v3 = [x[3] for x in v_keyed]
        self.vtypes = [x[1] for x in v_keyed]
        self.int_vtypes = [x[4] for x in v_keyed]

        if check and use_mmgroup_fast:
            def v3set(v3list):
                vlist = list(v3list)
                vlist += [gen_leech3_neg(v) for v in v3list]
                return set(gen_leech3_reduce(v) for v in vlist)

            m3 = MMOpFastMatrix(3, 4)
            m3.set_row(0, axis.v15 % 3)
            amod3 = MMOpFastAmod3(m3, 0)
            """
            amod3.raw_echelon(0)
            amod3.ker_img(mode_B = 0)
            assert amod3.len_B == (22, 2), amod3.len_B
            v3a = amod3.leech3vector(1, 22)
            v3b = amod3.leech3vector(1, 23)
            assert v3a > 0 and v3b > 0 and v3a != v3b
            vectors = [v3a, v3b, gen_leech3_add(v3a, v3b)]
            vectors.append(gen_leech3_add(v3a, vectors[-1]))
            assert v3set(vectors) == v3set(v3l)
            """
            buf = amod3.analyze_v4()
            assert buf[0] == self.v2[0]
            assert set(buf[1:3]) == set(self.v2[1:3])
            assert self.map_gcode_type(buf[3]) == self.vtypes[3], (buf[3], self.vtypes)
            assert set(self.v2list) == set(buf[4:])

    @property
    def v4list(self):
        if not self._v4list:
            self._v4list = []
            for i in range(100):
                v02 = self.v2list[i] ^ self._v2
                for j in range(i):
                     v = v02 ^ self.v2list[j]
                     if gen_leech2_type(v) == 4:
                         self._v4list.append(v)
            assert len(self._v4list) == len(set(self._v4list)) == 3850
        return self._v4list

    @property
    def v24list(self):
        return [x ^ self.v2[0] for x in self.v2list]


    @staticmethod
    def _subtypes_dict(v_list, subtypes):
        d = OrderedDict()
        for subt in subtypes:
            d[subt] = 0 
        for v in v_list:
            d[gen_leech2_subtype(v)] += 1
        return d

    @property
    def d2(self):
        if not self._d2:
            self._d2 = self._subtypes_dict(self.v2list, self.T2)
        return self._d2 

    @property
    def d24(self):
        return self._subtypes_dict(self.v24list, self.T4)

    @property
    def d4(self):
        if not self._d4:
            self._d4 = self._subtypes_dict(self.v4list, self.T4)
        return self._d4 

    def axis(self):
        """Return the axis stored in the object"""
        return Axis(self.g)

    def g_reduce_v3(self):
        """Return group element mapping this axis to a standard axis

        Here a standard axis is an axis referred by class StdAxis10A.
        """
        g_data = np.zeros(12, dtype = np.uint32)
        g_len = gen_leech3_reduce_leech_mod3(self.v3[3], g_data)
        return Xsp2_Co1('a', g_data[:g_len])

    def check_std_data(self):
        """Check that the data for this axis are consitent.

        We check that the 100 type-2 vectors for this axis computed
        by the mmgroup package are equal the 100 type-2 vectors
        computed by transforming the corresponding precomputed
        vectors for a standard axis.

        We also check the the scalar products with these 100 vectors
        with the decated vectors in member ``v2`` are as expected.
        """
        l2 = self.v2list
        g = self.g_reduce_v3() ** -1
        l2_ref = transform_fixed_leech2_set(StdAxis10A.v2list_C, g)
        assert set(l2_ref) == set(l2), (set(l2_ref), set(l2))
        for v2 in l2:
            assert gen_leech2_type(v2) == 2
 
        l23 = self.v2[:3]
        lscal = [4, 2, 2]
        l_type_233 = [2, 3, 3]
        for v23, ref, type_233 in zip(l23, lscal, l_type_233):
            assert gen_leech2_type(v23) == type_233
            for v2 in l2:
                assert gen_leech2_type(v2 ^ v23) == ref



class StdAxis10A:
    r"""Collect data for a standard axis of type 10A

    More specifically, a standard axis is a set of axes of type 10A
    that share a specific part :math:`300_x` in the vector
    representing the axis. This class stores a set of common
    properties of that part.

    Attribute ``v3[3]`` of of an instance of class ``Axis10A`` is a
    vector in the Leech lattice mod 3 that describes the part
    :math:`300_x` uniquely. C function ``gen_leech3_reduce_leech_mod3``
    defines a reduction process for such a vector in the Leech lattice
    mod 3. An axis of type 10A is reduced if attribute ``v3[3]`` of
    the corresponding instance of class ``Axis10A`` reduced as defined
    in that C function. For details we refer to
    Section *C interface for file gen_leech_reduce_mod3.c*
    in the document *The C interface of the mmgroup project*. 
    
    The following attributes are defined as the corresponding
    attributes or properties in class Axis10A:

    v3, v2, v2list, T2, T4.

    Attribute ``v2list_C`` is the list ``StdAxis10A.v2list`` encoded
    for use by function ``transform_fixed_leech2_set`` in
    module ``Case6A``.

    One cannot create instances of this class.
    """
    g_start, _ = parse_mat24_orbits("10A")
    axis_obj = Axis10A(Axis(g_start))
    del g_start
    g = axis_obj.g_reduce_v3()
    std_axis = axis_obj.axis() * g
    del g
    del axis_obj
    ad = Axis10A(std_axis)
    del std_axis
    ad.v2list.sort()
    v2list = ad.v2list
    v3 = ad.v3
    v2 = ad.v2
    v2list_C = py_prep_fixed_leech2_set(ad.v2list)
    T2 = ad.T2
    T4 = ad.T4
    del ad


#########################################################################
# Objective Golay code word
#########################################################################




def get_gc_objective(ad):
    data = []
    assert isinstance(ad, Axis10A)
    vtypes = ad.vtypes
    #print(vtypes)
    if vtypes[0] == "20":
        return [0]
    if vtypes[0] == "22":
        if vtypes[3] == "T13":
            return [(ad.v2[0] >> 12) & 0x7ff]
        return [0]
    assert vtypes[0] == "21"
    for i in range(1, 3):
        if vtypes[i] == "34":
             data.append((ad.v2[i] >> 12) & 0x7ff)
    v3 = ad.v3[3]
    #print(hex(v3))
    supp = (v3 ^ (v3 >> 24)) & 0xffffff
    syndromes =  mat24.all_syndromes(supp)
    for syn in syndromes:
        t = supp ^ syn
        #if mat24.bw24(t) in [8,16]:
        #    data.append(mat24.vect_to_gcode(t) & 0x7ff)
    if len(data):
        return data
    # yet to be finished!
    return data


def check_gc_objective(ad, gc_list):
    if len(gc_list) == 0:
        print("WTF!!!")
        return 
    gc_dict = {}
    for gc in gc_list:
        gc_dict[gc & 0x7ff] = 0
    v4list = [x for x in ad.v4list if (x & 0x800) == 0]
    for v in v4list:
        gc = (v >> 12) & 0x7ff
        if gc in gc_dict:
            gc_dict[gc] += 1
    for gc in gc_dict:
        if gc_dict[gc] == 0:
            print("WTF0!!!", len(gc_list), gc_dict)
    print(list(gc_dict.values()))

def process_gc_objective(ad):
    gc_list = get_gc_objective(ad)
    check_gc_objective(ad, gc_list)


#########################################################################
# Auxiliary functions for the strategy in the subsequent subsections
#########################################################################


max_buf_size = 0

def store_buffer_size(*args):
    global max_buf_size
    buf_size = 0
    for arg in args:
        if isinstance(arg, dict):
            buf_size += sum(len(x) for x in arg.values())
        elif isinstance(arg, list):
            buf_size += len(arg)
        else:
            buf_size += arg
    max_buf_size = max(max_buf_size, buf_size)

def gcode(v):   
    return (v & 0x800) + ((v >> 12) & 0x7ff)



#########################################################################
# The strategy for a dedicated type-2 vector of subtype 21
#########################################################################



def check_strategy21(ad, verbose = 0):
    assert isinstance(ad, Axis10A)
    v2 = ad.v2[0]
    pool1, pool2 = [], []
    for v in ad.v2list:
        if v & 0x800 == 0:
            pool1.append(v)
        else:
            vv = v ^ v2 
            if gen_leech2_subtype(vv) in [0x42, 0x44]:
                pool2.append(vv)
    if verbose:
        print("pools", [len(x) for x in [pool1, pool2]])
    store_buffer_size(pool1, pool2)
    for w2 in pool2:
        succ = 0   
        for w1 in pool1:
             succ |=  gen_leech2_subtype(w1 ^ w2) in [0x42, 0x44]
        if not succ:
            raise ValueError("strategy21 fails")
    if verbose:
        print("strategy suceeds")



def strategy21(ad):
    v2 = ad.v2[0]
    pool, v_min = [], 0x1000000
    for v in ad.v2list:
        if v & 0x800 == 0:
            pool.append(v)
        else:
            vv = v ^ v2 
            if vv < v_min:
                if gen_leech2_subtype(vv) in [0x42, 0x44]:
                    v_min = vv
    assert v_min < 0x1000000
    assert len(pool) <= 66
    pool.sort()
    for v in pool:
        v1 = v ^  v_min
        if gen_leech2_subtype(v1) in [0x42, 0x44]:
            return v1
    raise ValueError("strategy 21 has failed")


#########################################################################
# The strategy for a dedicated type-2 vector of subtype 22
#########################################################################





def check_strategy22_T13(ad, verbose = 0):
    #print("X", [hex(x) for x in ad.v2], [x for x in ad.vtypes])
    ref_list = []
    for vtype, vt3 in zip(ad.vtypes, ad.v2):
        if vtype == '34':
            ref_list.append(vt3  & 0x7ff800)
    assert len(ref_list) == 1, len(ref_list)
    vlist = []
    for v in ad.v2list:
        if v & 0x7ff800 == ref_list[0]:
            vlist.append(v)
    assert len(vlist) == 2, len(vlist)
    v_result = vlist[0] ^ vlist[1] ^ ad.v2[0]
    assert gen_leech2_subtype(v_result) == 0x42



def check_strategy22(ad, verbose = 1):
    v2 = ad.v2[0]
    d = defaultdict(list)
    d2 = defaultdict(list)
    m = (int(ad.vtypes[1]) & 1) ^ 1
    for v in ad.v2list:
        if  ((v & 0x800) >> 11) ^ m:
            vv = v ^ v2 
            d[gcode(v)].append(v)
            d2[gcode(vv)].append(vv)
    store_buffer_size(d, d2)
    s1 = set(d.keys())
    s2 = set(d2.keys())
    isect = set(x for x in d if x in d2)
    assert len(isect) > 0

    if verbose:
        l1 = [len(x) for x in d.values()]
        l2 = [len(x) for x in d2.values()]
        print("l", l1, sum(l1))
        print("l2", [l2], sum(l2))
        print(len(isect))

             

def strategy22_T13(ad):
    v_ref = None
    for vtype, vt3 in zip(ad.vtypes, ad.v2):
        if vtype == '34':
            v_ref = gcode(vt3)
    vlist = []
    for v in ad.v2list:
        if gcode(v) == v_ref:
            vlist.append(v)
    result = vlist[0] ^ vlist[1] ^ ad.v2[0]
    assert gen_leech2_subtype(result) == 0x42
    return result

def strategy22(ad):
    d = defaultdict(lambda : [[],[]])
    v2 = ad.v2[0]
    m = (int(ad.vtypes[1]) & 1) ^ 1
    store_buffer_size(ad.v2list)
    for v in ad.v2list:
        if  ((v & 0x800) >> 11) ^ m:
            vv = v ^ v2 
            d[gcode(v)][0].append(v)
            d[gcode(vv)][1].append(vv)
    ld = min([x for x in d if len(d[x][0]) and len(d[x][1])])
    ld1, ld2 = d[ld]
    result = min(ld1) ^ min(ld2)
    assert result & 0x7ff800 == 0
    return result


#########################################################################
# The strategy for a dedicated type-2 vector of subtype 20
#########################################################################



def check_strategy20(ad, verbose = 0):
    v2 = ad.v2[0]
    d = defaultdict(list)
    store_buffer_size(ad.v2list)
    type7 = ad.vtypes[3]
    if type7 == 'U7':
        assert 0x800000 in ad.v4list
        return
    m = type7 != 'S22'
    for v in ad.v2list:
        if  ((v & 0x800) >> 11) ^ m:
            d[gcode(v)].append(v)
    l_gt1 = [len(x) for x in d.values() if len(x) > 1]
    if verbose:
        print("l", l_gt1, sum(l_gt1))
    assert len(l_gt1)

def strategy20(ad):
    d = defaultdict(list)
    v2 = ad.v2[0]
    type7 = ad.vtypes[3]
    if type7 == 'U7':
        return 0x800000
    m = type7 != 'S22'
    for v in ad.v2list:
        if  ((v & 0x800) >> 11) ^  m:
            d[gcode(v)].append(v)
    l_min = min([x for x in d if len(d[x]) > 1])
    ld = d[l_min]
    ld.sort()
    result = ld[0] ^ ld[1] ^ v2
    assert result & 0x7ff800 == 0
    return result





#########################################################################
# The general strategy
#########################################################################

def check_strategy(ad, verbose = 0):
    v2 = ad.v2[0]
    t = ad.vtypes[0]
    if t == '21':
       check_strategy21(ad, verbose)
    elif t == '22':
        if  ad.vtypes[3] == 'T13':
            check_strategy22_T13(ad, verbose)
        else:
            check_strategy22(ad, verbose)
    elif t == '20':
       check_strategy20(ad, verbose)
    else:
       raise ValueError("Bad type-2 vector")


def strategy(ad):
    v2 = ad.v2[0]
    t = ad.vtypes[0]
    if t == '21':
        return strategy21(ad)
    elif t == '22':
        if  ad.vtypes[3] == 'T13':
            return strategy22_T13(ad)
        else:
            return strategy22(ad)
    elif t == '20':
        return strategy20(ad)
    else:
        raise ValueError("Bad type-2 vector")



#########################################################################
# Check transformation with type-4 vector
#########################################################################


def check_transformation(axis, v, verbose = 0):
    axis1 = axis * Xsp2_Co1('c', v)**-1
    weights_BpmC = axis_count_BpmC(axis1)
    assert set(weights_BpmC) == {22,181}
    if verbose:
        print("Weight of part B +- C", weights_BpmC)
        for i in range(3):
            axis1.display_sym(i, mod=3)
    e = weights_BpmC.index(181) + 1
    assert axis1.axis_type(e) == '4B'


def check_C_transformation(axis, v4):
    """Check C program for computing type-4 vector for a 10A axis

    The function uses the C functions exported as methods of the
    Cython class ``MMOpFastMatrix`` for computing a type-4 vector
    to be used for 'reducing' axis ``axis``.

    The function fails if that type-4 vectors differs from vector
    ``v4``. The pythonic way to compute ``v4`` is:

    ``ad = Axis10A(axis); v4 = strategy(ad)``

    The function performs no action of the ``mmgrpup_fast``
    package is not present.
    """
    if use_mmgroup_fast:
        m3 = MMOpFastMatrix(3, 4)
        m3.set_row(0, axis.v15 % 3)
        amod3 = MMOpFastAmod3(m3, 0)
        t, v4_c = amod3.find_v4()
        assert t == "10A", t
        #print(i+1, hex(v4), hex(v4_c))
        assert v4 == v4_c, (hex(v4), hex(v4_c))

#########################################################################
# Display reduction information for the case 6A
#########################################################################




def compute(ntests = 10, verbose = 0):
    g_start, orbits = parse_mat24_orbits("10A")
    data = []
    for g in orbits:
        axis = Axis(g_start * g)
        axis *= MM('r', 'N_x0')
        ad = Axis10A(axis)
        data.append(
           [ad.vtypes, list(ad.d4.values()), list(ad.d2.values()),
               list(ad.d24.values()), ad])
    data.sort(key = lambda a: (a[0][0], a[0][3], a[0]))
    for i, d in enumerate(data):
        #if d[0][0] != "21": continue
        s0, s1, s2, s3 = map(str, d[0])
        l1, l2, l3 = str(d[1][:4]), str(d[2]), str(d[3][:4]) 
        print(
        f"{i+1:2}: {s0:2} {s1:2} {s2:2} {s3:3}  {l1:19} {l2:12} {l3:15}"
        ) 
        ad = d[-1] 
        check_strategy(ad)
        for j in range(ntests):
            axis1 = ad.axis() * MM('r', 'N_x0')
            ad = Axis10A(axis1)
            if j & 7 == 1:
                ad.check_std_data()
            v = strategy(ad)
            assert 0 < v <= 1 << 24, v
            check_transformation(axis1, v)
            check_C_transformation(axis1, v)


def compute_general(ntests = 1000, verbose = 0):
    g_start, _ = parse_mat24_orbits("10A")
    start_axis = Axis(g_start)
    for n in range(ntests):
        axis = start_axis * MM('r', 'G_x0')
        ad = Axis10A(axis)
        v = strategy(ad)
        assert 0 < v <= 1 << 24, v
        check_transformation(axis, v, verbose)
        if n % 1000 == 999:
            print(".", end = "", flush = True)
        check_C_transformation(axis, v)

    print("\nGeneral test of axis case 10A passed")
    print("Buffer size required:", max_buf_size)    



#########################################################################
# Test reduction for the case 10A1
#########################################################################



def strategy_100(ad):
    MAP_V = {0x48:1, 0x40:2, 0x42:3, 0x44:3, 0x46:4, 0x43:5}
    vlist = ad.v24list
    vlist1 = [(MAP_V[gen_leech2_subtype(v)], v) for v in vlist]
    return min(vlist1)[1]    
        




def test_case_10A1(ntests = 1000, verbose = 0):
    g_start, _ = parse_mat24_orbits("10A")
    start_axis = Axis(g_start)
    for n in range(ntests):
        axis = start_axis * MM('r', 'G_x0')
        ad = Axis10A(axis)
        m3 = MMOpFastMatrix(3, 4)
        m3.set_row(0, axis.v15 % 3)
        amod3 = MMOpFastAmod3(m3, 0)
        v2 = randint(1, 0xffffff)
        t, v4_c = amod3.find_v4(v2)
        assert t == "10A", t
        v4_ref = strategy_100(ad)
        assert v4_c == v4_ref, (hex(v4_c), hex(v4_ref))
        axis1 = axis * Xsp2_Co1('c', v4_c)**-1
        weights_BpmC = axis_count_BpmC(axis1)
        assert set(weights_BpmC) == {1, 232}
        e = weights_BpmC.index(1) + 1
        assert axis1.axis_type(e) == '6A', axis1.axis_type(e)
        if verbose:
            print("Weight of part B +- C", weights_BpmC)
            for i in range(3):
                axis1.display_sym(i, mod=3)

    print("\nGeneral test of axis case 10A1 passed")




#########################################################################
# Tables for code generation
#########################################################################




class Tables:
    tables = {
        "MM_AXIS3_CASE10A_VTYPE2": StdAxis10A.v2list_C
    }
    directives = {
    }


#########################################################################
# Display reduction information for the case 6A
#########################################################################


def test_10A_all(n_tests = 100):
    std_v2list_C = StdAxis10A.v2list_C
    if use_mmgroup_fast:
        compute(ntests = 1 + n_tests // 20, verbose = 0)
        compute_general(ntests = n_tests, verbose = 0)
        test_case_10A1(ntests = n_tests, verbose = 0)

if __name__ == "__main__":
   test_10A_all()

