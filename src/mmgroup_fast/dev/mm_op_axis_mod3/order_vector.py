from collections import defaultdict
import numpy as np


from mmgroup import MMV, MM0, MM, MMSpace




def make_A12():
    # Has yet errors!!!!!!!!!!!
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
    print(A12)
    print(np.linalg.det(A12))
    print([x for x in data_a24()])
    
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
