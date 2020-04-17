import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from graphviz import Digraph
import logging
logger = logging.getLogger(__name__)

from IR import pb_to_ir
from checker import operator_list


def run_all_pass(graph):
    logger.info("run all passes...")

    # passes.eliminate_dropout.run(graph)
    # passes.eliminate_identity.run(graph)
    # passes.eliminate_pad.run(graph)

    # passes.convert_constant_to_init.run(graph)
    # passes.convert_shape_to_init.run(graph)
    # passes.convert_gather_to_init.run(graph)
    # passes.convert_unsuqeeze_to_init.run(graph)
    # passes.convert_concat_to_init.run(graph)
    # passes.convert_flatten_to_reshape.run(graph)
    # passes.convert_reduceMean_to_globalAveragePool.run(graph)

    # passes.fuse_pad_into_averagePool.run(graph)
    # passes.fuse_pad_into_maxPool.run(graph)
    # passes.fuse_pad_into_conv.run(graph)


    return graph




if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnx_model  output_model")
        sys.exit(-1)

    # pb_to_ir
    graph = pb_to_ir.convert(sys.argv[1])
    graph.dump()

    # optimize the grapg
    graph = run_all_pass(graph)
  
    # ir_to_pb
    onnx_model = ir_to_pb.convert(graph)
    print('save onnx model ...')
    onnx.save(onnx_model, sys.argv[2]+".onnx")
    
