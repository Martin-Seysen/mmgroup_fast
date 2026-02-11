from collections import defaultdict
import numpy as np


from mmgroup import MMV, MM0, MM, MMSpace




def make_A12():
    a = np.zeros((12, 12), dtype = np.int32)
    MARKED = {0,1,2,3,4,7}
    for i in range(12): a[i, i] = 1 if i in MARKED else -1
    for i in range(1, 12): a[0, i] = a[i, 0] = 1
    data = {
        1: [8,9,10,11],
        2: [8,9,10],
        3: [8,9],
        4: [8],
    }
    for i, row in data.items():
        for y in row:
            sign = 1 if y >= 0 else -1
            j = abs(y)
            a[i, j] = a[j, i] = sign
    det = np.linalg.det(a)
    idet = int(np.round(det))
    assert abs(det - idet) < 1.0e-8
    assert idet % 3 != 0
    return a

A12 = make_A12()

def display_a12():
    print("Submatrix of part A of vector v_0")
    print(A12)
    print("Determinant of submatrix is %.3f" %
        np.linalg.det(A12))
    #print([x for x in data_a24()])
    
MAP_A12 = [0,1,2,3,4,5,6,8,9,10,12,16]
KER_A12 = [7]

def data_a24():
    for i in range(12):
        ii = MAP_A12[i]
        for j in range(i, 12):
            jj, e = MAP_A12[j], A12[i,j]
            if e:
                 yield (int(e) % 3, 'A', ii, jj)
    neg_diag = set(MAP_A12) | set(KER_A12)
    for i in range(24):
        if i not in neg_diag:
            yield (2, 'A', i, i)


def process_A24(verbose = 0):
    A24 = np.zeros((24,24), dtype = np.uint32)
    for x, tag, i, j in data_a24():
        if tag == "A":
           A24[i,j] = A24[j,i] = x
    assert (A24[7] == 0).all()
    A23 = np.copy(A24)
    A23 = np.delete(A23, 7, 0)
    A23 = np.delete(A23, 7, 1)
    if verbose:
        print("Part A of vector v_0 (modulo 3)")
        print(A24)
        #print(A)
    det = np.linalg.det(A23)
    idet = int(np.round(det))
    assert abs(det - idet) < 1.0e-5
    assert idet % 3 != 0
    if verbose:
         print("Determinant of image of that part is %.3f"
             % np.linalg.det(A23))
    d, n = {}, 0
    for i in range(24):
        if A24[i,i] < 2:
            d[(int(A24[i,i]), int(np.count_nonzero(A24[i])))] = i
        else:
            n += 1
    assert set(d.values()) == set([0,1,2,3,4,7,8])
    assert len(d) + n == 24
    if verbose:
         print("Hash values for part A:")
         print(d)
    return  A24, d


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


if __name__ == "__main__":
    display_a12()
    process_A24(verbose = 1)
