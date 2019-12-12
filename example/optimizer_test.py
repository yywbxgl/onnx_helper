import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx

from IR import pb_to_ir
from IR import ir_to_pb
from optimizer import operator_convert

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnx_model  output_model")
        sys.exit(-1)

    # pb_to_ir
    graph = pb_to_ir.convert(sys.argv[1])
    pb_to_ir.dump(graph)

    graph = operator_convert.run_pass(graph)
  
    # pb_to_ir
    onnx_model = ir_to_pb.convert(graph)
    print('save onnx model ...')
    onnx.save(onnx_model, sys.argv[2]+".onnx")