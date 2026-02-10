"""Register custom markers for pytest.

This module registers the custom markers for pytest so 
that we may invoke:

pytest --pyargs mmgroup [options]

without 'PytestUnknownMarkWarning' warnings. 
This tests the **installed** mmgroup package.

This registering is usually done in the configuration 
file pytest.ini in the root directory, see:

https://pytest.org/en/7.4.x/reference/customize.html#pytest-ini

Alternatively, we may register these markers in file conftest.py.
Here file conftest.py may be loacated in the  installed
python package mmgroup, see:

https://pytest.org/en/7.4.x/how-to/writing_plugins.html#registering-custom-markers

"""


# Registration of markers in the style used by file pytest.ini 
markers = r"""
   basic:       very basis tests
   bench:       benchmark
   compiler:    test requires a (gcc) compiler
   mm_op:       test for python module mm_op
   alignment:   test alignment issues
   mm_amod3:    test for python module mm_op_axis_mod3
   user:        interaction tests that should be done by the user
   slow:        marks tests as slow (deselect with '-m "not slow"')
"""

# Hook for extending pytest configuration
def pytest_configure(config):
    for line in markers.split("\n"):
        if len(line) and not line.isspace():
            config.addinivalue_line("markers", line.strip())



