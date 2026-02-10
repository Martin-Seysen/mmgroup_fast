"""Semi-automatic generation of constants for Hadamard matrix"""

from numbers import Integral
from mmgroup.bitfunctions import bitparity, bitweight


def write_list(l, end = True, braces = 1, llen = 16, indent = 4):
    """Format a list as a constant to be written into a C program

    Here ``l`` is a list of integers or strings containing the data of
    the constant to be generated. The generated C constant is enclosed
    in ``braces`` opening and ``braces`` closing curly braces. Each
    output line contains ``llen`` entries. If ``end`` is False then a
    comma is appended to the constant. Each output line is indented 
    by ``indent`` blank characters.
    """
    data = []
    openb, closeb, subindent = "{" * braces, "}" * braces, " " * braces 
    for i in range(0, len(l), llen):
        part = ",".join([("%2d" % x) if isinstance(x, Integral) else x
            for x in l[i : i + llen]])
        if i + llen < len(l):
            part += ","
        else:
            part += closeb if end else closeb + ","
        part = (openb if i == 0 else subindent) + part
        data.append(" " * indent + part)
    return "\n".join(data) + "\n"
    
        

def write_hadamard_shuffle_list(n, braces = 1, indent = 4):
    for k in range(n):
        l = [i ^ (1 << k) for i in range(1 << n)]
        s = write_list(l, k == n-1, braces, llen = 16, indent = indent)  
        print(s)


def write_hadamard_neg_list(n, braces = 1, indent = 4):
    for k in range(n):
        l = [bool(i & (1 << k)) for i in range(1 << n)]
        l1 = ["0F"[x] for x in l]
        s = write_list(l1, k == n-1, braces, llen = 32, indent = indent)  
        print(s)


def write_hadamard_swap_parity_list(n, braces = 1, indent = 4):
    nn = 1 << n
    l = [nn - 1 - i if  bitparity(i) else i for i in range(1 << n)]
    s = write_list(l, True, 1, llen = 16, indent = indent)  
    print(s)


def write_hadamard_inv_swap_parity_list(n, braces = 1, indent = 4):
    nn = 1 << n
    l = [i if  bitparity(i) else nn - 1 - i for i in range(1 << n)]
    s = write_list(l, True, 1, llen = 16, indent = indent)  
    print(s)



if __name__ == "__main__":
    N = 5
    INDENT = 10
    write_hadamard_shuffle_list(N, braces = 2, indent = INDENT)
    print("")
    write_hadamard_neg_list(N, braces = 2, indent = INDENT)
    print("")
    write_hadamard_swap_parity_list(N, braces = 1, indent = INDENT)
    print("")
    write_hadamard_inv_swap_parity_list(N, braces = 1, indent = INDENT)


