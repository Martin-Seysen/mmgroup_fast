import numpy as np
from mmgroup.bitfunctions import bin
from mmgroup.clifford12 import bitmatrix64_mul

####################################################################
####################################################################
# 23-bit LFSR with taps 19 and 21
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
        M1 = bitmatrix64_mul(M2, M2)
    if verbose:
        print("Bit matrix to the power of 2**%d" % e)
        for i in range(DIM):
            print(" ",  bin(M1[i], DIM, 8, reverse=True))
        print()
    return M1


def make_start_array24(verbose=0):
    a = np.zeros((24,32), dtype = np.uint8)
    m = make_mat_lfsr()
    for col in range(32):
        for sh in range(6, 8):
            for row in range(23):
                a[row,col] |= ((m[0] >> row) & 1) << sh
            ms = mat_lfsr_repeated_square(m, 14)
            mc = np.copy(m)
            m = bitmatrix64_mul(mc, ms)
    a[23] = a[23-23] ^ a[23-18]
    if verbose:
        print("LFSR start values for filling order vector")
        for row in range(24):
            for col in range(32):
                x = a[row][col]
                assert x & 0x3f == 0
                print("%2d" % (x >> 6), end = "")
            print("")
    return a


def make_start_array6(verbose=0):
    a24 = make_start_array24()
    a = np.zeros((6,32), dtype = np.uint8)
    for i in range(6):
        for j in range(4):
            a[i] ^= a24[4*i + j] >> (2*j)
    if verbose:
        s = "LFSR start values for filling order vector compressed (hex)"
        print(s)
        for row in range(6):
            for col in range(32):
                x = a[row][col]
                print(" %02x" % x, end = "")
            print("")
    return a
    

####################################################################
####################################################################
# Table for code generator
####################################################################
####################################################################


class Tables:
    tables = {
        "MM_AXIS3_START_VECTOR_LFSR": make_start_array6(),
    }
    directives = {
    }



####################################################################
####################################################################
# Main program for testing
####################################################################
####################################################################



def display_all():
    make_start_array24(verbose=1)
    make_start_array6(verbose=1)

if __name__ == "__main__":
    display_all()

