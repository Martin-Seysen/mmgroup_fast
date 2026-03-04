from collections import defaultdict
import numpy as np


from mmgroup import MMV, MM0, MM, MMSpace, mat24
from mmgroup.clifford12 import bitmatrix64_mul
from mmgroup.clifford12 import bitmatrix64_vmul

####################################################################
####################################################################
# 21-bit LFSR with taps 19 and 21
####################################################################
####################################################################

# Dimension and tap positions of LFSR
DIM, TAPS = 23, [23, 18]

def make_mat_lfsr(verbose = 0):
    M = np.zeros(DIM, dtype = np.uint64)
    for i in range(DIM - 1):
         M[i+1] = 1 << i
    for t in TAPS:
        M[DIM - t] |= 1 << (DIM - 1)
    if verbose:
        from mmgroup.bitfunctions import bin
        print("Bit matrix for %d-bit LFSR" % DIM)
        for i in range(DIM):
            print(" ",  bin(M[i], DIM, 8, reverse=True))
        print()
    return M

def mat_lfsr_repeated_square(M, e, verbose = 0):
    """Return M ** (2**e)"""
    M1 = np.copy(M)
    for i in range(e):
        M2 = np.copy(M1)
        M1 = bitmatrix64_mul(M1, M2)
    if verbose:
        from mmgroup.bitfunctions import bin
        print("Bit matrix to the power of 2**%d" % e)
        for i in range(DIM):
            print(" ",  bin(M1[i], DIM, 8, reverse=True))
        print()
    return M1

def mat_V(verbose = 0):
    M = make_mat_lfsr()
    Me = mat_lfsr_repeated_square(M, 14)
    Me = bitmatrix64_mul(Me, M)
    COLS = 32
    A = np.zeros((DIM, COLS), dtype = np.uint8)
    v = 0x7ab3d5
    #print(M.shape, Me.shape)
    for i in range(COLS):
        for sh in range(8):
            v = bitmatrix64_vmul(v, Me, DIM)
            for j in range(DIM):
                 A[j,i] |= ((v >> j) & 1) << sh
    if verbose:
        print("Subblock of generated LFSR table:")
        print(A[5:15, 10:20])


def display_bitmatrix(verbose = 1):
    M = make_mat_lfsr(verbose)
    mat_lfsr_repeated_square(M, 12, verbose)
    mat_V(verbose)



####################################################################
####################################################################
# Composing the vector v_0 with the axcs v^+ and v^-
####################################################################
####################################################################



class MM_Matrix:
    ALIST = [(1, 'A', 2, 2), (1, 'A', 3, 3), (2, 'A', 2, 3)] 
    def __init__(self):
        self.d = defaultdict(lambda : [None, None, None, None])

    def add_entry(self, row, value, tag, *indices):
        v = MMSpace.index_to_sparse(tag, *indices)
        value %= 3
        #print(row, value, tag, indices, hex(v), self.d[v])
        assert self.d[v][row] in [None, value]
        self.d[v][row] = value;

    def add_entry_list(self, row, data_list):
        for data in data_list:
            self.add_entry(row, *data)

    def get_array(self):
        a = np.zeros(len(self.d), dtype = np.uint32)
        for i, ind in enumerate(sorted(self.d.keys())):
            data = [x if x else 0 for x in self.d[ind]]
            v = sum(x << 2*j for j, x in enumerate(data))
            a[i] = ind + v
        return a

    def store_to_matrix(self, m=None):
        from mmgroup_fast.mm_op_fast import MMOpFastMatrix
        if m is None:
            m = MMOpFastMatrix(3, 4, 1)
        assert isinstance(m, MMOpFastMatrix)
        m.zero_data()
        a = self.get_array()
        m.put_data(a)
        return m

    def std_matrix(self):
        for i in [0,1]:
            self.add_entry_list(i, self.ALIST)
            self.add_entry(i, i+1, 'B', 2, 3)
        return self.store_to_matrix()



def std_matrix():
    return MM_Matrix().std_matrix()


####################################################################
####################################################################
# Main program for testing
####################################################################
####################################################################



def display_all():
    #display_a12()
    #A, sp, h, y = process_A24(verbose = 1)
    x_tuples = [t for t in data_BCTX()]
    ReduceGx0Data.display()
    #display_bitmatrix(verbose = 1)
    #print("Matrix A is sparse notation:")
    #for i, e in enumerate(sp):
    #    print("%07x" % e, end = "\n" if i % 8 == 7 else " ")
    #print()

if __name__ == "__main__":
    display_bitmatrix(verbose = 1)

