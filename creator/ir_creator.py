import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from IR import ir_config
from IR import ir

def parse_config(ir_config):
    f = open(ir_config, "r")
    line = f.readline()
    if "input" not in line:
        print("can not find graph input")
    temp = line.split(":")[1].split("  ")
    name = temp[0]
    dims = eval(temp[1])
    print(type(dims))
    print(name, dims)


def create_graph(ir_config):
    f = open(ir_config, "r")
    f_lines = f.readlines()
    graph = ir.Graph


if __name__ == "__main__":
    parse_config("test.cfg")
    