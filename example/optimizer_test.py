import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from onnx import optimizer
from onnx import helper, shape_inference
import numpy as np
import onnxruntime.backend as backend
import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
# fmt = "%(filename)s:%(lineno)d %(levelname)s - %(message)s"
coloredlogs.install(level="INFO", fmt=fmt)
# coloredlogs.install(level="DEBUG", fmt=fmt)
logger = logging.getLogger(__name__)

from IR import pb_to_ir
from IR import ir_to_pb
from optimizer import operator_convert
from checker import onnx_check
# from optimizer.onnxsim import onnx_simplifier
from simplifyer import onnx_simplifier


if __name__ == "__main__":

    if len(sys.argv) <2:
        print ("Usage:", sys.argv[0], "onnx_model  [output_model]")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = input_file.split(".onnx")[0] + "_optimized.onnx"
    if (len(sys.argv) == 3):
        output_file = sys.argv[2]

    # ---- step0. simplify onnx model
    onnx_ori = onnx.load(input_file)
    onnx_sim = onnx_simplifier.simplify(input_file, input_shape=[1,224,224,3])
    onnx.save(onnx_sim, output_file)

    # ---- step1. optimize ir graph
    # pb_to_ir
    graph = pb_to_ir.convert(output_file)
    graph.dump()

    # ir graph optimize
    graph = operator_convert.run_all_pass(graph)
    # graph.dump()
  
    # pb_to_ir
    onnx_model = ir_to_pb.convert(graph)
    onnx.save(onnx_model, output_file)

    # test
    onnx_simplifier.test_conveted_model(onnx_ori, onnx_model)
    
    # ---- step4.check operator support
    ret = onnx_check.ir_op_check(graph)
    if ret == False:
        logger.warn("onnx operator check not pass!")
    else:
        logger.info("check pass")

    # -----setp 5 compile onnx_model, save loadable
    logger.info("change version to 3/8")
    onnx_model.ir_version=3
    onnx_model.opset_import[0].version = 8
    file_name = output_file + "onnx_ir3.onnx"
    loadable_name = output_file + "onnx_ir3.nbdla"
    logger.info('save onnx model %s ...', file_name)
    onnx.save(onnx_model, file_name)
    onnx_simplifier.test_conveted_model(onnx_ori, file_name)

    complier = "../tools/ys11_bin_onnc.nv_large  "
    cmd = complier + file_name + "  -o " + loadable_name
    os.system(cmd)

