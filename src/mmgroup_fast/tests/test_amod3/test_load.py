from collections import OrderedDict
import numpy as np
from random import randint, shuffle, sample

import pytest

from mmgroup_fast.mm_op_fast import MMOpFastMatrix, MMOpFastAmod3
from mmgroup import MM0, MMV, MM, XLeech2, Cocode


MMV3 = MMV(3)


def row_to_leech3(row):
     D = [0, 1, 0x1000000, 0]
     return sum(D[row[i] & 3] << i for i in range(24))



@pytest.mark.mm_amod3
def test_load_amod3():
    for i in range(2):
        a = MMOpFastMatrix(3,4)
        vectors = {}
        for j in range(4):
            v = MMV3('R')
            vectors[j] = v
            a.set_row(j, v)
        for j in range(4):
            a3 = MMOpFastAmod3(a, j)
            data = a3.raw_data[0]
            assert ((data & 0xfc) == 0).all()
            assert (data % 3 == vectors[j]['A']).all()
            row = randint(0, 23)
            v3ref = row_to_leech3(a3.raw_data[0, row])
            v3 = a3.leech3vector(0, row)
            assert v3 == v3ref, (hex(v3), hex(v3ref), a3.raw_data[0, row])


def echelon_mod_p(a, diag = 0, p = 3):
    nrows, ncols = a.shape 
    a1 = np.zeros((nrows, nrows + ncols), dtype = np.uint32)
    a1[:, :ncols] = a % p
    if diag:
        f = -diag % p
        a1[:, :ncols] += f * np.eye(nrows, ncols, dtype = np.uint32)
    a1[:, ncols:] = np.eye(nrows, dtype = np.uint32)
    a1 %= p
    row = 0
    for col in range(ncols):
        for row1 in range(row, nrows):
            if a1[row1, col]:
                a1[[row,row1]] = a1[[row1,row]]
                piv = ((p - a1[row, col]) * a1[row]) % p
                for row2 in range(row+1, nrows):
                    a1[row2] += piv * a1[row2, col]
                    a1[row2] %= p
                row += 1
                break
    return row, a1[:, :ncols], a1[:, ncols:]



def echelon_amod3_testcases():
    data = [np.eye(24), np.eye(24)]
    data[1][1,0] = data[1][1,3]  = 2
    data[1][1,1] = 0
    for m in data:
        yield MMOpFastAmod3(m, 0), 0
    for i in range(3):
        for d in range(3):
            yield MMOpFastAmod3('r', 0), d


@pytest.mark.mm_amod3
def test_echelon_amod3(verbose = 0):
    for n, (a, d) in enumerate(echelon_amod3_testcases()):
        if verbose:
            print("\nTest", n+1)
        ref_row, ref_img, ref_ker = echelon_mod_p(a.data[0], d)
        if verbose:
            print("")
            print("data0\n", a.data[0])
            print("ref_img\n", ref_img)
        row, ech = a.raw_echelon(d)
        if verbose:
            print("raw data\n", a.raw_data[1])
        assert (ech & 0xcc == 0).all()
        img = (ech & 3) % 3
        ker = ((ech >> 4) & 3) % 3
        if verbose:
            print(img)
        for i in range(24):
            if (img[i] != ref_img[i]).any():
                print("!", i); break
        assert row == ref_row
        assert (img == ref_img).all()
        assert (ker == ref_ker).all()



class AxisInvariant:
    def __init__(self, axis):
        self.name = axis.axis_type()
        a = MMOpFastAmod3(axis)
        self.norm = a.norm
        self.invar = []
        for d in range(3):
            dim_img, _ = a.raw_echelon(d)
            assert a.mode_B == 4 
            assert a.dim_img == dim_img
            a_copy = MMOpFastAmod3(a)
            img, ker = a_copy.ker_img(0) 
            a_copy = MMOpFastAmod3(a)
            assert (a.raw_data == a_copy.raw_data).all()
            _, isect = MMOpFastAmod3(a_copy).ker_img(2) 
            self.invar.append((len(ker), len(isect)))
            # More_stuff is yet to be tested!!!
    @property
    def as_str(self):
        k0, k1, k2 = [ "%2d" % x for x, _ in self.invar]
        s0, s1, s2 = [ " *!"[min(x,2)]  for _, x in self.invar]
        s = f"{self.name:3}   {self.norm:4}    {k0}{s0}   {k1}{s1}   {k2}{s2}  {self.invar}"
        return s

    def sort_key(self):
        return self.norm, self.invar, int(self.name[:-1]), self.name

    @classmethod
    def headline(self):
        return "name  norm  ker0  ker1  ker2"



@pytest.mark.mm_amod3
def test_display_axis_invariants(verbose = 0):
    print("\nInvariants of part 'A' of axes (mod 3)")
    print(AxisInvariant.headline())
    axes = []
    from mmgroup.axes import Axis
    for axis in Axis.representatives().values():
        a = AxisInvariant(axis.v15)
        axes.append((a.sort_key(), a))
    for (_, a) in sorted(axes):
        print(a.as_str)
    if verbose:
        y = Axis.representatives()['6C']
        y.display_sym(mod=3, text ='Case 6C')
        for i in range(20):
            cc = Cocode(sample([0,4,5,6,7], 4))
            g = MM('c', XLeech2(0, cc))
            y1 = y * g**-1
            for e in (1,):
                print(e, y1.axis_type(e))
    

