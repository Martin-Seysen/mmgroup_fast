
try:
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
    use_mmgroup_fast = True
except:
    print("Package mmgroup_fast not found")
    use_mmgroup_fast = False
    # raise # We can't do this in the code generation phase!!!


