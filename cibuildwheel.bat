pip uninstall mmgroup_fast
python -m cibuildwheel  --output-dir wheelhouse --only cp311-win_amd64
pip install wheelhouse/mmgroup_fast-0.0.0-cp311-cp311-win_amd64.whl

