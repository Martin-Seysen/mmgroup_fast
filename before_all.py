import os
import sys
import shutil


def before_all():
    import mmgroup
    lib_dir = os.path.split(mmgroup.__file__)[0]
    libs = ["libmmgroup_mat24"]
    dest_dir = os.path.join("src", "mmgroup")
    #os.makedirs(real_path, exist_ok=True)
    for lib in libs:
        shutil.copyfile(os.path.join(lib_dir, lib),
            os.path.join(dest_dir, lib))


if __name__ == "__main__":
   before_all()