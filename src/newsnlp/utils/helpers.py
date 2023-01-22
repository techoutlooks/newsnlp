import os


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

