import os


def get_asset_path(filename):
    cwd = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(cwd, "assets", filename)
    if not os.path.exists(fp):
        fp = os.path.join(cwd, "..", "assets", filename)
    if not os.path.exists(fp):
        raise FileNotFoundError("Asset file {} does not exist".format(filename))
    return fp
