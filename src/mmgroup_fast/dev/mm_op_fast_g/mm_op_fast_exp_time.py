
BASE_TIMES = {
   4: (3,0,0,0),
   5: (4,0,0,0),
   6: (2,1,1,1),
   7: (3,1,1,1)
}

def exp_time_std(e):
    assert e >= 4
    bpos = e.bit_length() - 1
    d = (e >> (bpos - 2)) & 3
    nshort, nlong, nreduce, has3 = BASE_TIMES[d+4]
    bpos -= 2
    while bpos > 0:
        k = min(bpos, 2)
        if k == 2:
            d =  (e >> (bpos - 2)) & 3;
            if d == 2:
                k = d = 1;
        else:
            d =  (e >> (bpos - 1)) & 1;
        nreduce += 1
        nlong += (1 << k) - 1
        if d == 1:
            nshort += 1
        if d == 3:
            nlong += 1
            if not has3:
                nreduce += 1
                has3 = 1
        bpos -= k 
    return nshort, nlong, nreduce



def exp_time(e):
    if 1 <= e < 6:
        return e-1, 0, 0
    if e == 0:
        return 0, 0, 0
    if e < 0:
        s, l, r = exp_time(-e)
        return s+1, l, r
    if e % 4 == 0:
        s, l, r = exp_time(e // 4)
        return 3, s+l, r+1
    if e % 3 == 0:
        s, l, r = exp_time(e // 3)
        return 2, s+l, r+1
    return exp_time_std(e) 
        


def display_std_effort(n):
    for e in range(4, n+1):
        print("%2d" % e, exp_time_std(e), exp_time(e))


if __name__ == "__main__":
    display_std_effort(100)
  
    
