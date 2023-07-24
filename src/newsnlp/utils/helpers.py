"""
FIXME: replace with ad.py from latest version
"""
import json
import os
import pickle
from typing import Tuple, Any
from subprocess import run as _run, PIPE


# run cmd in subshell
run = lambda cmd: _run(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)


def get_path_stats(path: str) -> Tuple[dict, str]:
    """ Get size and file count of directory """

    size = run(f"du -hs {path}")
    file_count = run(f"ls -al {path}|wc -l")
    return {
        'size': size.stdout.split('\t')[0],
        'file_count': file_count.stdout.split()[0],
    }, size.stderr or file_count.stderr


def get_env_variable(name, *args) -> str:
    assert len(args) < 2, "`get_env_variable()` supports at most one positional argument " \
                          "for the environment variable's default value."
    try:
        return os.environ[name]
    except KeyError:
        if len(args):
            return args[0]
        message = "Expected environment variable '{}' not set.".format(name)
        raise Exception(message)


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def save_fixture(path: str, data: Any, use_pickle=False):
    pickler, mode = (pickle, "wb") if use_pickle else (json, "w")
    with open(path, mode) as fp:
        pickler.dump(data, fp)


def load_fixture(path, use_pickle=False):
    pickler, mode = (pickle, "rb") if use_pickle else (json, "r")
    with open(path, mode) as fp:
        return pickler.load(fp)
