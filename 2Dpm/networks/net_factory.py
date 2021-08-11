import importlib


def get_network(name):
    m = importlib.import_module("networks.{}".format(name))
    return m.model
