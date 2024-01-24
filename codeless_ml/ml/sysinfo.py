import sys

import tensorflow as tf

if __name__ == "__main__":
    print(f"tensorflow version {tf.__version__}")
    print(f"python version: {sys.version}")
    print(f"{tf.sysconfig.get_build_info()}")
