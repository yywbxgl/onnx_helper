import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from graphviz import Digraph

from IR import pb_to_ir
from IR import ir_to_pb
from checker import operator_list
from optimizer import passes


def run_pass(graph):
    #todo 根据命令选择不同的优化case进行运行
    # graph = eliminate_node.run(graph, "Dropout")

    return graph


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnx_model  output_model")
        sys.exit(-1)

    from optimizer import eliminate_node

    # pb_to_ir
    graph = pb_to_ir.convert(sys.argv[1])
    pb_to_ir.dump(graph)

    graph = eliminate_node.run(graph, "Dropout")
  
    # pb_to_ir
    onnx_model = ir_to_pb.convert(graph)
    print('save onnx model ...')
    onnx.save(onnx_model, sys.argv[2]+".onnx")
    
