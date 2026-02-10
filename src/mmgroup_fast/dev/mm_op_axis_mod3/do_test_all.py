
def test_all(n_tests = 50):
    from mmgroup_fast.dev.mm_op_axis_mod3.Case2B import test_2B_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case4C import test_4C_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6A import test_6A_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6F import test_6F_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case8B import test_8B_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case10A import test_10A_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case10B import test_10B_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case12C4B import test_12C4B_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case6C import test_6C_all
    from mmgroup_fast.dev.mm_op_axis_mod3.Case4A import test_4A_all
    test_4A_all(n_tests)
    test_6C_all(n_tests)
    test_12C4B_all(n_tests)
    test_2B_all(n_tests)
    test_4C_all(n_tests)
    test_6A_all(n_tests)
    test_6F_all(n_tests)
    test_8B_all(n_tests)
    test_10A_all(n_tests)
    test_10B_all(n_tests)



#########################################################################
# Main program
#########################################################################



if __name__ == "__main__":
    try:
       n_tests = int(sys.argv[1])
    except:
       n_tests = 50
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "..","..","..")
    sys.path.append(os.path.abspath(path))
    test_all(n_tests)

