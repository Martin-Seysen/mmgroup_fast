import sys
import os
import re
from collections import defaultdict, OrderedDict
import numpy as np
from mmgroup import MM, XLeech2, mat24, GcVector, GCode, AutPL, Xsp2_Co1
from mmgroup.generators import gen_leech2_type
from mmgroup.generators import gen_leech2_subtype, gen_leech3_op_vector_word
from mmgroup.generators import gen_leech3_op_vector_atom
from mmgroup.generators import gen_leech3_reduce, gen_leech3to2_short
from mmgroup.generators import gen_leech3_add, gen_leech2to3_abs, gen_leech3_neg
from mmgroup.generators import gen_leech3_reduce_leech_mod3, gen_leech3_neg
from mmgroup.generators import gen_leech2_op_word
from mmgroup.generators import gen_leech2_op_word_leech2_many
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_echelon_eqn
from mmgroup.clifford12 import leech2_matrix_orthogonal
from mmgroup.clifford12 import leech2_matrix_radical
from mmgroup.clifford12 import leech2matrix_add_eqn
from mmgroup.clifford12 import leech2matrix_subspace_eqn
from mmgroup.clifford12 import bitmatrix64_vmul
from mmgroup.axes import Axis
from mmgroup.mm_reduce import mm_reduce_analyze_2A_axis

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))


try:
    from mmgroup_fast import MMOpFastMatrix, MMOpFastAmod3
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_prep_fast_transform_fix_leech2
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_map_Case6A
    from mmgroup_fast.mm_op_fast import mm_axis3_fast_find_case_2A
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False


#########################################################################
# Auxiliary functions
#########################################################################

def mul_v3_g(v3, g):
    """Return v3 * g for a Leech lattice vector v3 mod 3""" 
    v3 = gen_leech3_op_vector_word(v3, g.mmdata, len(g.mmdata)) 
    return gen_leech3_reduce(v3)




def abs_equ_v3(v3a, v3b):
    """Return True if v3a == v3b (up to sign)

    Here v3a, v3b are vectors in the Leech lattice mod 3
    """ 
    v3a = gen_leech3_reduce(v3a)
    v3b = gen_leech3_reduce(v3b)
    v3bn = gen_leech3_reduce(gen_leech3_neg(v3b))
    return v3a in [v3b, v3bn]


#########################################################################
# Parse the Mat_24 of an axis on a G_x0 orbit
#########################################################################


def download_github_file(repo_url, file_path_in_repo, local_path):
    """
    Download a file from a GitHub repo and save it locally.
    
    :param repo_url: URL of the GitHub repo (e.g. https://github.com/user/repo)
    :param file_path_in_repo: Path to the file inside the repo (e.g. folder/file.txt)
    :param local_path: Local file path to save the downloaded file
    """
    try:
        import requests
        # Convert GitHub repo URL to raw content URL
        if repo_url.endswith('/'):
            repo_url = repo_url[:-1]
        user_repo = repo_url.replace("https://github.com/", "")
        
        raw_url = f"https://raw.githubusercontent.com/{user_repo}/main/{file_path_in_repo}"
        
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Write content to local file
        with open(local_path, "wb") as f:
            f.write(response.content)
        
        print(f"File downloaded and saved to: {local_path}")
    
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")


REPO_URL = "https://github.com/Martin-Seysen/order_monster"
REPO_PATH = "axis_orbits/certificates/axis_certificate.txt"
LOCAL_PATH = "./axis_certificate.txt"



def open_axis_certificate(update = False):
    """Return axis certificate from github repo as file object

    The name of the grithub repo is:
    "https://github.com/Martin-Seysen/order_monster"

    The file object is obtained from a local copy (if present),
    unless parameter 'update' is True.
    """ 
    if update or not os.path.isfile(LOCAL_PATH):
        import requests
        download_github_file(REPO_URL, REPO_PATH, LOCAL_PATH)
    return open(LOCAL_PATH, "rt")


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



def parse_mat24_orbits(orbit_name):
    read_orbits = False
    orbits = []
    with open_axis_certificate() as f:
        for line in f:
            tag, value, g = parse_line(line)
            if tag == 'axis' and value == orbit_name:
                g_start = MM(g)
                read_orbits = True
            if tag == 'end':
                read_orbits = False
            if read_orbits and tag == 'orb':
                orbits.append(MM(g))
    return g_start, orbits



#########################################################################
# Return a vector in Lambda mod 3 of type 6_22 describing a 6A orbit
#########################################################################

def axis_to_MMOpFastMatrix(axis):
    m3 = MMOpFastMatrix(3, 4)
    m3.set_row(0, axis.v15 % 3)
    amod3 = MMOpFastAmod3(m3, 0)
    return amod3


def x_equations_6A(axis, v2_ref = 0, verbose = 0):
    if not use_mmgroup_fast:
        return x_equations_6A_mmgroup(axis)
    amod3 = axis_to_MMOpFastMatrix(axis)
    amod3.raw_echelon(0)
    v_out_buf = [int(x) for x in MMOpFastAmod3(amod3).analyze_v4()]
    assert v_out_buf[0] == 6, v_out_buf[0]
    v_out = v_out_buf[2] + (v_out_buf[3] << 24) + (v_out_buf[1] << 48)
    #v_out = MMOpFastAmod3(amod3).analyze_case_6A()
    #print("vvv", v_out)
    amod3.ker_img(mode_B = 0)
    n_BC = amod3.num_entries_BC()
    assert amod3.len_B == (1,23), amod3.len_B
    v3 = amod3.leech3vector(1, 0)
    assert v3 > 0
    v2 = gen_leech3to2_short(v3)
    if v2_ref:
        assert v2 == v2_ref.ord & 0xffffff, (hex(v2), hex(v3), hex(v2_ref.ord))
    nrows = 0
    subspace64 = np.zeros(24, dtype = np.uint64)
    subspace = np.zeros(24, dtype = np.uint64)
    ortho_compl = np.zeros(24, dtype = np.uint64)
    nrows += leech2matrix_add_eqn(subspace64, nrows, 24, v2)
    v_scalar = v2 + 0x2000000
    for i in range(100):
        v_new = amod3.rand_short_nonzero(v_scalar)
        nrows += leech2matrix_add_eqn(subspace64, nrows, 24, v_new)
        if nrows >= 11:
            break
    v_scalar = v2
    for i in range(100):
        v_new = amod3.rand_short_nonzero(v_scalar)
        nrows += leech2matrix_add_eqn(subspace64, nrows, 24, v_new)
        if nrows >= 22:
            break
    #print("nrows =",  nrows, "n_BC =", amod3.num_entries_BC())
    leech2matrix_echelon_eqn(subspace64, nrows, 24, subspace);  
    m = leech2_matrix_orthogonal(subspace, ortho_compl, nrows)
    assert m == 22
    v2_other = ortho_compl[22]
    if v2_other == v2:
        v2_other = ortho_compl[23]
     
    v3_other = gen_leech2to3_abs(v2_other)
    assert v3_other > 0, hex(v3_other)
    for i in range(2):
        v3_other = gen_leech3_add(v3_other, v3)
        support = (v3_other ^ (v3_other >> 24)) & 0xffffff
        weight = mat24.bw24(support)
        if weight % 3 == 0:
            coc_weight = mat24.cocode_weight(mat24.vect_to_cocode(support))
            if verbose:
                print("%012x %2d %1d" % (v3_other, weight, coc_weight))
            assert v_out == v3_other + (weight << 48), (hex(v_out), hex(v3_other), weight) 
            return v3_other, weight, n_BC
    raise ValueError("No suitable vector mod 3 found!") 



#########################################################################
# count nonzero entries of parts B + C and B - C of an axis (mod3)
#########################################################################

def axis_count_BpmC(axis):
    assert isinstance(axis, Axis)
    B = axis.v15['B'] % 3
    C = axis.v15['C'] % 3
    p = np.count_nonzero((B+C) % 3)
    m = np.count_nonzero((B+15-C) % 3)
    return int(p) >> 1, int(m) >> 1

#########################################################################
# Substitute for function x_equations_6A not using mmgroup_fast
#########################################################################


V3_0 = 0xe # The type 6_22 vector in Leech lattice mod 2 of the
           # representative of Axis type 6A in its G_x0 orbit.

# V3_0 = 0x200000c # a bad candidate for V3_0 !


def x_equations_6A_mmgroup(axis):
    g = axis.reduce_G_x0()
    gi = g**-1
    v3 = gen_leech3_op_vector_word(V3_0, gi.mmdata, len(gi.mmdata))
    supp = (v3 ^ (v3 >> 24)) & 0xffffff
    w = mat24.bw24(supp)
    B = (axis.v15['B'] % 3 != 0).ravel()
    C = (axis.v15['C'] % 3 != 0).ravel()
    n_BC = sum(list(B) + list(C))
    return v3, w, n_BC

#########################################################################
# Transform fixed set of vectors in Leech lattice mod 2
#########################################################################


def check_fixed_leech2_set(vlist, v):
    lv2, lv1 = divmod(v[0], 0x100)
    assert len(v) == lv1 + lv2 + 1
    v1 = v[1:lv1+1]
    ind2 = v[lv1+1:]
    a = np.array(v1, dtype = np.uint64)
    v1 = [int(v) for v in v1]
    v2 = [bitmatrix64_vmul(i, a, len(a)) for i in ind2]
    ref_set = set(v1 + v2)
    test_set = set(int(v) for v in vlist)
    assert test_set == ref_set


def py_prep_fixed_leech2_set(vlist):
    vl = np.zeros(len(vlist) + 1, dtype = np.uint32)
    a = np.zeros(24, dtype = np.uint64)
    nrows = lv2 = 0
    for v in vlist:
        v &= 0xffffff
        if leech2matrix_add_eqn(a, nrows, 24, v): 
            nrows += 1
            vl[nrows] = v
        else:
            lv2 += 1
            vl[-lv2] = v
    vl[0] = nrows + (lv2 << 8);
    ps = vl[nrows + 1:]
    for i, v in enumerate(ps):
        x = leech2matrix_subspace_eqn(a, nrows, 24, v)
        assert x >= 0
        ps[i] = x
    return vl
    


def py_transform_fixed_leech2_set(fixed_set, g):
    m, n = divmod(fixed_set[0], 0x100)
    gm = g.mmdata
    v = np.zeros(m + n, dtype = np.uint32)
    v[:n] = fixed_set[1:n+1]
    assert gen_leech2_op_word_leech2_many(v, n, gm, len(gm), 0) == 0
    av = np.array(v[:n], dtype = np.uint64)
    for i, x in enumerate(fixed_set[n+1:n+1+m]):
        v[n+i] = bitmatrix64_vmul(x, av, n)
    return v;
        


def prep_fixed_leech2_set(vlist, check = True):
    if not use_mmgroup_fast:
        return py_prep_fixed_leech2_set(vlist)
    vl = np.array(vlist, dtype = np.uint32)
    v = np.zeros(len(vlist) + 1, dtype = np.uint32)
    mm_axis3_prep_fast_transform_fix_leech2(vl, len(vl), v)
    if check:
        check_fixed_leech2_set(vlist, v)
        assert (v == py_prep_fixed_leech2_set(vlist)).all()
    return v

def transform_fixed_leech2_set(fixed_set, g, check=True):
    v = np.zeros(100, dtype = np.uint32)
    m = g.mmdata
    status = mm_axis3_fast_transform_fix_leech2(
        fixed_set, m, len(m), v, len(v))
    assert status >= 0, status
    v = v[:status]
    if check:
        v_ref = py_transform_fixed_leech2_set(fixed_set, g)
        assert set(v_ref) == set(v)
    return v
   

#########################################################################
# Make dictionary for reducing weight of negative entries in lambda mod 3
#########################################################################

def make_case_weight12():
    lst = []
    for i in range(0, 12, 4):
        for j in range(1, 4):
            v = (1 << i) + (1 << (i+j)) 
            v0 = v + (1 << 16) + (1 << 20)
            for k in range(1,4):
                v1 = v0 ^ (1 << (16 + k))
                syn = mat24.syndrome(v1)
                if (syn & 0xfff) == 0:
                    #print(i, j, hex(v), hex(v1), hex(syn))
                    lst.append((i+j, v1 ^ syn))
                    break
    assert len(lst) == 9, len(lst)
    lst.append((0, 0xe11111))
    return lst



def make_case_weight6():
    d = {}
    for i in range(759):
        o = mat24.octad_to_vect(i)
        if o & 1:
           o1 = o ^ 1
           o8 = o1 & 0xff
           if (o8 & (o8 - 1)) == 0:
               b = mat24.lsbit24(o1)
               if b not in d:
                   d[b] = o
    lst =  [(b, d[b]) for b in [1,2,3,5,6,7]]
    lst.sort()
    #print(lst)
    return lst
           
def make_case_weight15():
    lst = []
    for i in (12, 16, 20):
        for j in (1, 2, 3):
            tet = (1 << i) + (1 << (i+j)) + (1 << (j+8)) + (1 << 8)
            o = (tet ^ 1) ^ mat24.syndrome(tet ^ 1)
            assert mat24.bw24(o & 0xffff00) == 4
            lst.append((i+j, o))
    lst.append((20, 0x11111e)) 
    lst.append((11, 0xf0f))
    return lst 

CASE_WEIGHTS = defaultdict(list)

def case_weights():
    global CASE_WEIGHTS
    if len(CASE_WEIGHTS) == 0:
       CASE_WEIGHTS[12] = make_case_weight12()
       CASE_WEIGHTS[6] = make_case_weight6()
       CASE_WEIGHTS[3] = CASE_WEIGHTS[6][:3]
       CASE_WEIGHTS[18] = [(8, 0xffff00), (4, 0xff)]
       CASE_WEIGHTS[15] = make_case_weight15()
    return CASE_WEIGHTS
    
def reduce_neg_weight(v3, weight):
    supp = (v3 ^ (v3 >> 24)) & 0xffffff
    v_neg = (v3 >> 24) & supp
    y = 0
    for w, yw in case_weights()[weight]:
        vw = 1 << w
        if v_neg & vw:
            y ^= yw; v_neg = (v_neg ^ yw) & supp           
    return y
  

#########################################################################
# Map an axis to its representative in its N_x0 orbit
#########################################################################

d_std_Nx0_img_p = {}

def add_std_Nx0_img_p(weight, image):
    global d_std_Nx0_img_p
    if weight not in d_std_Nx0_img_p:
        d_std_Nx0_img_p[weight] = image 

TAG_Y = 0x40000000
TAG_P = 0x20000000

def map_pi_y(v3):
    g, len_g = np.zeros(3, dtype = np.int32), 0
    def mul_g(atom):
       nonlocal v3, g, len_g
       v3 = gen_leech3_op_vector_atom(v3, atom)
       g[len_g] = atom
       len_g += 1
    def mul_y(y):
        mul_g(TAG_Y + GCode(y).ord)
    supp = GcVector((v3 ^ (v3 >> 24)) & 0xffffff)
    synd = supp.all_syndromes()
    neg = GcVector(~v3 & (v3 >> 24))
    zerobit = min(mat24.lsbit24(~supp.ord & 0xffffff), 23)
    y = GcVector(neg + neg.syndrome(zerobit))
    mul_y(y)
    pi = AutPL(0)
    w = len(supp)
    if w ==24:
        neg = GcVector(~v3 & (v3 >> 24))
        syn = neg.syndrome(0)
        blist = syn.bit_list
        img = [0,4]
    elif w == 3:
        blist = supp.bit_list
        img = [1,2,3]
    elif w in [6, 18]:
        if len(supp) == 18:
           supp = ~supp
        blist = supp.bit_list
        img = [1,2,3,5,6,7]
    elif w == 12:
        blist = []
        for syn in synd:
            if syn & supp == syn:
                if len(blist):
                    blist.append(mat24.lsbit24(syn.ord))
                else:
                    blist = syn.bit_list
        img = [0,1,2,3,4,8]
    elif w == 15:
        syn = synd[0]
        octad = ~(supp + syn)
        assert len(octad) == 8
        assert 1 << mat24.lsbit24(syn.ord) == syn.ord
        blist = octad.bit_list[:6] + syn.bit_list[:1]
        img = [0,1,2,3,4,5,8]
    else:
        raise ValueError("Illegal weight of Leech lattice vector mod 3")
    add_std_Nx0_img_p(w, img)
    pi = AutPL(0, zip(blist, img), 0)    
    mul_g(0x20000000 + pi.perm_num)
 
    y = GcVector(reduce_neg_weight(v3, w)) 
    mul_y(y)
    #if mat24.bw24(v3 >> 24) > mat24.bw24(v3 & 0xffffff):
    #    v3 = gen_leech3_reduce(gen_leech3_neg(v3))
    return v3, MM('a', g)
  

#########################################################################
# Dictionary for type-4 vectors for representatives of N_x0 orbits
#########################################################################



TYPES4 = [0x48, 0x40, 0x42, 0x44, 0x46, 0x43]

def get_type_2_4_vectors_6A(axis, check=True):
    buf = np.zeros(900, dtype = np.uint32)
    assert mm_reduce_analyze_2A_axis(axis.v15.data, buf) == 0
    assert buf[0] == 0x61, buf[0]
    assert buf[3] == 892, (buf[3], buf[890:])
    assert gen_leech2_type(buf[4]) == 2
    v_out = buf[5:5+891]
    assert len(v_out) == 891
    for v in v_out:
        assert gen_leech2_type(v) == 4
    return buf[4], v_out


def analyze_suborbits_6A(axis):
    suborbits = OrderedDict()
    for t in TYPES4:
        suborbits[t] = 0
    v2, v4list = get_type_2_4_vectors_6A(axis)
    for v in v4list:        
        suborbits[gen_leech2_subtype(v)] += 1
    return v2, suborbits


def best_type4_vectors_6A(axis):
    suborbits = OrderedDict()
    for t in TYPES4:
        suborbits[t] = []
    _, v4list = get_type_2_4_vectors_6A(axis)
    for v in v4list:
        suborbits[gen_leech2_subtype(v)].append(v)
    for t in TYPES4:
        if len(suborbits[t]):
            suborbits[t].sort()
            return suborbits[t]  
    raise ValueError("No type-4 vectors for case 6A found")


d_std_Nx0_t4_vect = {}


def add_std_Nx0_t4_vect(axis, v3, check=True):
    global d_std_Nx0_t4_vect
    v3 = gen_leech3_reduce(v3)
    supp = (v3 ^ (v3 >> 24)) & 0xffffff
    weight = mat24.bw24(supp)
    best = best_type4_vectors_6A(axis)
    best_encode = prep_fixed_leech2_set(best)
    if not weight in d_std_Nx0_t4_vect:
        d_std_Nx0_t4_vect[weight] = (v3, best_encode, best)
    elif check:
        v3_ref, encode_ref, best_ref = d_std_Nx0_t4_vect[weight]
        assert v3 == v3_ref, (hex(v3), hex(v3_ref))
        if list(best) != list(best_ref):
            ERR = "Inconsstent orbits of 6A axis given by v3, weight = %d"
            print([hex(x) for x in best])
            print([hex(x) for x in best_ref])
            raise ValueError(ERR % weight)
        assert list(best_encode) == list(encode_ref)


def v3_transform_type4_list(v3, check = True):
    v3 = gen_leech3_reduce(v3)
    supp = (v3 ^ (v3 >> 24)) & 0xffffff
    weight = mat24.bw24(supp)
    v3_reduced, g = map_pi_y(v3)
    v3_rep, best_rep, v4_std = d_std_Nx0_t4_vect[weight]
    assert v3_reduced == v3_rep
    best = transform_fixed_leech2_set(best_rep, g**-1)
    if check:
        best_ref = np.copy(v4_std)
        gen_leech2_op_word_leech2_many(best_ref, len(best_ref), 
            g.mmdata, len(g.mmdata), True)
        assert set(best) == set(best_ref)
    return best

def v3_transform_type4(v3, check = True):
    return min(v3_transform_type4_list(v3, check))


#########################################################################
# Auxiliary functions for displaying reduction information
#########################################################################

def mul_leech2_g(v2, g):
    g.reduce()
    assert g.in_G_x0()
    mm = g.mmdata
    return gen_leech2_op_word(v2 & 0xffffff, mm, len(mm)) & 0xffffff

def mul_leech3_g(v3, g):
    g.reduce()
    assert g.in_G_x0()
    mm = g.mmdata
    return gen_leech3_op_vector_word(v3, mm, len(mm))



def weight_mod3(v3):
    v3 = gen_leech3_reduce(v3)
    w3 =  (v3 ^ (v3 >> 24)) & 0xffffff
    return GcVector(w3).vtype()

  



#########################################################################
# Display reduction information for the case 6A
#########################################################################



def display_stat_6A(ntests = 10, check = True, verbose = True):
    suborbit_list = []
    g_start, orbits = parse_mat24_orbits("6A")
    d0, d1, d2 = [XLeech2(0, c) for c in ([2,3], [1,2], [1,3])]
    V3 = 0xe  # extra vector mod 3 in subspace spanned by d0, d1, d2 
    for g in orbits:
        for i in range(ntests):
            g *= MM('r', 'N_x0')
            axis = Axis(g_start * g)
            v2, suborbits = analyze_suborbits_6A(axis)
            data = [suborbits[key] for key in TYPES4]
            short = gen_leech2_subtype(v2)
            assert short == (d0 * g).xsubtype
            other_shorts = [(d * g).xsubtype for d in (d1, d2)]
            other_shorts.sort()
            other_types = [axis.axis_type(e) for e in (1,2)]
            other_types.sort(key = lambda s: (int(s[:-1]),s))
            weight3 =  weight_mod3(mul_leech3_g(V3, g))
            v3, weight, n_BC = x_equations_6A(axis, v2_ref = d0 * g)
            v3a, g1 = map_pi_y(v3)
            add_std_Nx0_t4_vect(axis * g1, v3a)
            g1i = g1**-1 
            v3a_std, best_std_enc, best_std = d_std_Nx0_t4_vect[weight]
            ref_suborbits = [mul_leech2_g(x, g1i) for x in best_std]
            _, all_orbits = get_type_2_4_vectors_6A(axis)
            st = gen_leech2_subtype(ref_suborbits[0])
            suborbits = [x for x in all_orbits
               if gen_leech2_subtype(x) == st]
            assert set(ref_suborbits) <= set(suborbits)

            if check and use_mmgroup_fast:
                v4 = v3_transform_type4(v3)
                axis1 = axis * Xsp2_Co1('c', v4)**-1
                red_types = [axis1.axis_type(e) for e in (1,2)]
                assert set(red_types) == set(['4A', '6A'])
                buf_g = np.zeros(3+77, dtype = np.uint32)
                l = mm_axis3_fast_map_Case6A(v3, buf_g)
                assert l >= 0, l
                best = buf_g[3:3+l]
                g2i = MM('a', buf_g[:3])
                g2 = g2i ** -1
                assert g1 == g2
                tf = transform_fixed_leech2_set(
                   d_std_Nx0_t4_vect[weight][1], g2i)
                assert (tf == best).all()
                assert v4 == min(best)
                assert axis_to_MMOpFastMatrix(axis).find_v4() == ('6A', v4)
                t_markers = axis_count_BpmC(axis1)
                assert set(t_markers) == {1, 231}
                e = t_markers.index(231) + 1
                assert axis1.axis_type(e) == '4A', axis1.axis_type(e)

            if i == 0 and verbose:
                suborbit_list.append(
                    (short, other_shorts, other_types,
                        data, weight3, v3a, n_BC)
                )
    if verbose:
        subtypes_list = [hex(x)[2:] for x in TYPES4]
        print("Orbit statistics, subtypes = ", subtypes_list)
        FMT ="%02x (%02x,%02x), %-26s, weight=%4s, %012x; n_BC=%3d"
        suborbit_list.sort()
        for entry in suborbit_list:
            short, others, types, data, weight3, v3a, n_BC = entry
            data = [short] + others + [data, weight3, v3a, n_BC]
            print (FMT % tuple(data))



#########################################################################
# Tables for C version
#########################################################################

"""Data structures in C file are

typedef struct  {
    uint8_t  weight;      // key for structure: weight of support of v3
    uint8_t  len_img_p;   // length of image of permutation p
    uint8_t  start_img_p; // start of image of permutation p
    uint8_t  len_op_y;    // length of y operation table
    uint8_t  start_op_y;  // start of y operation table
    uint8_t  start_v4;    // start of encoded list of type-4 vectors
    uint64_t v3_ref;      // expected value of v3 after applying p and y
} info_6A_type;

uint8_t a_img_p[];  // array of concatenated images of perm. p 

uint32_t a_op_y[];  // array of concatenated y operations

uint32_t a_v4[];    // array of encoded type-4 vectors

Entries of ``a_op_y`` are 32-bit integers encoded as follows:

Bits 31,...,24:  Encode a bit position. If v3 has a negative entry
                 at that position then it must be transformed with
                 MM('y', y). 

Bits 23,...,0:   The Golay cod word y in ``vector`` representation.

Array ``a_v4`` with entries starting at bit postion given by component
``start_v4`` in an encoded array of vectors in the Leech lattice mod 2,
for use with function ``mm_axis3_fast_transform_fix_leech2``.
"""

W_KEYS = [3, 6, 12, 15, 18, 24]

def make_table():
    display_stat_6A(ntests = 1, verbose = False) 
    info_6A = []
    a_img_p = []
    a_op_y = []
    a_v4 = []
    for w in W_KEYS:
        entry = [w]
        img_p = d_std_Nx0_img_p[w] if w in d_std_Nx0_img_p else []
        entry += [len(img_p), len(a_img_p)]
        a_img_p += img_p

        op_y = CASE_WEIGHTS[w] if w in CASE_WEIGHTS else []
        op_y = [(i << 24) + v for i, v in op_y]
        entry += [len(op_y), len(a_op_y)]
        a_op_y += op_y

        v3, l_v4, _ = d_std_Nx0_t4_vect[w]
        entry += [len(a_v4), v3]
        a_v4 += list(l_v4) 
        info_6A.append(entry)  
    return info_6A, a_img_p, a_op_y, a_v4


class Tables:
    info_6A, a_img_p, a_op_y, a_v4 = make_table()
    tables = {
        "MM_AXIS3_CASE6A_INFO_6A": info_6A,
        "MM_AXIS3_CASE6A_A_IMG_P": a_img_p,
        "MM_AXIS3_CASE6A_A_OP_Y": a_op_y,
        "MM_AXIS3_CASE6A_A_V4": a_v4,
    }
    directives = {
    }



#########################################################################
# Check case 2A
#########################################################################


def iter_testcases_2A(n_tests=10):
    std_axis = Axis(MM())
    std_invol = XLeech2(0x200)
    amod3 = axis_to_MMOpFastMatrix(std_axis)
    for i in range(3):
        g = Xsp2_Co1('r', 'N_x0')
        yield std_axis * g, std_invol * g
    for i in range(n_tests):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis * g, std_invol * g



def find_case_2A(v2, v2_main = 0):
    status = mm_axis3_fast_find_case_2A(v2, v2_main)
    assert status > 0, status
    return divmod(status, 0x2000000)

def check_case_2A(n_tests=10):
    INVOL_ST = {0x21:0x42, 0x22:0x40}
    if use_mmgroup_fast:
        for axis, invol in iter_testcases_2A(n_tests):
            amod3a = axis_to_MMOpFastMatrix(axis)
            t, inv = amod3a.find_v4()
            assert inv == invol.ord
            st = gen_leech2_subtype(inv)
            assert t == "2A"
            mode, v_op = find_case_2A(inv)
            assert mode & ~1 == 0
            assert (inv | v_op) & ~0x1ffffff == 0
            #print("2A", hex(inv), mode, hex(v_op), hex(st))
            if mode:
                assert st in [0x21, 0x22]
                assert gen_leech2_subtype(v_op) == INVOL_ST[st]
                assert gen_leech2_type(v_op ^ inv) == 2 
            else:
                assert inv == v_op, (hex(inv), hex(v_op), st)
                assert st == 0x20




def iter_testcases_2A_baby(n_tests=10):
    std_axis = Axis(MM())
    std_invol = XLeech2(0x200)
    invol_v2a = XLeech2(0x800200)
    invol_v2b = XLeech2(0x1000200)
    amod3 = axis_to_MMOpFastMatrix(std_axis)
    for i in range(n_tests):
        g = Xsp2_Co1('r', 'N_x0')
        yield std_axis * g, std_invol * g, (invol_v2a * g).ord
        yield std_axis * g, std_invol * g, (invol_v2b * g).ord
    for i in range(n_tests):
        g = Xsp2_Co1('r', 'G_x0')
        yield std_axis * g, std_invol * g, (invol_v2a * g).ord
        yield std_axis * g, std_invol * g, (invol_v2b * g).ord


def check_case_2A_baby(n_tests=10):
    if use_mmgroup_fast:
        for axis, invol, v2_main in iter_testcases_2A_baby(n_tests=10):
            amod3a = axis_to_MMOpFastMatrix(axis)
            t, inv = amod3a.find_v4()
            assert inv == invol.ord
            assert t == "2A"
            mode, v_op = find_case_2A(inv, v2_main)
            assert mode & ~1 == 0
            assert (inv | v_op) & ~0x1ffffff == 0
            #print("2A%d" % (1-mode), hex(inv), hex(v2_main), hex(v_op))
            if mode:
                gen_leech2_type(v_op) == 4 
                gen_leech2_type(v_op ^ inv) == 2 
            else:
                assert (inv ^ v2_main) == 0x1000000


  
#########################################################################
# Main program
#########################################################################

def test_6A_all(n_tests=10):
    check_case_2A(n_tests)
    check_case_2A_baby(n_tests)
    display_stat_6A(ntests = max(5, 1 + n_tests//3))


if __name__ == "__main__":
    test_6A_all()



