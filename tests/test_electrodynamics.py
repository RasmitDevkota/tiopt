import ctypes
import pathlib

libname = pathlib.Path().absolute() / "../libtiopt.so"
libtiopt = ctypes.CDLL(libname)



