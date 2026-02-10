"""Executable program for class generate_code.CodeGenerator"""

import sys


if __name__ == "__main__":
    sys.path.append('src')
    from mmgroup_fast.dev.mm_op_fast.mm_op_fast_xi_monomial import analyze_tables
    analyze_tables()
