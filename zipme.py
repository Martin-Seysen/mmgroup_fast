"""Zip the whole project to a zip file in subdirectory 'build'.


The zip process is based on:

python setup.py sdist --format=zip

Non-source files are removed from the zip file.

"""


from __future__ import absolute_import, division, print_function
from __future__ import  unicode_literals


import zipfile, time
import sys, os
import subprocess
import shutil

from zipfile import ZIP_DEFLATED


tm = time.strftime("%Y_%m_%d_%H_%M")

filename = r'mmgroup_fast_%s.zip' % tm


#source = r"dist\mmgroup_fast-0.0.0.zip"

zip_path = os.path.join('backup', filename)

#backup_dir = r"backup"


def make_archive():
    shutil.make_archive(
        os.path.splitext(zip_path)[0],
        'zip',
        root_dir = '..',
        base_dir = 'mmgroup_fast/src'
    )
    

def add_archive():
    with zipfile.ZipFile(zip_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for filename in files:
             #print(filename)
             data = open(filename, 'rb').read()
             out_filename = os.path.join('mmgroup_fast', filename)
             zf.writestr(out_filename, data)


if __name__ == '__main__':
    subprocess.check_call([sys.executable, 'cleanup.py', '-pxcs'])
    make_archive()
    add_archive()
