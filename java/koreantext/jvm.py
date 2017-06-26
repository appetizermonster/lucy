from os import path
from jpype import *
import sys
import re


def init_jvm():
    if isJVMStarted():
        return

    jvm_path = getDefaultJVMPath()

    # HACK fixing jvm path for darwin
    if sys.platform == 'darwin' and jvm_path.endswith('libjvm.dylib'):
        jvm_path = re.sub(r'/lib/.+/libjvm.dylib',
                          '/lib/jli/libjli.dylib', jvm_path, 0)

    cur_dir = path.dirname(__file__)
    jars_path = path.abspath(path.join(cur_dir, 'jars'))
    startJVM(jvm_path,
             '-Djava.ext.dirs=%s' % jars_path,
             '-Dfile.encoding=UTF8')
