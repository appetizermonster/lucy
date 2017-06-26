from jpype import *


def to_jstring(str):
    return java.lang.String(str)
