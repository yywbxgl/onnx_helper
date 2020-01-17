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
from optimizer.onnxsim import onnx_simplifier

def onnx_optmize(model_path):

    original_model = onnx.load(model_path)

    all_passes = optimizer.get_available_passes()
    logger.info("Available optimization passes:")
    for p in all_passes:
        logger.debug(p)
    logger.debug('---------')

    passes = []
    # passes.append('eliminate_unused_initializer')
    # passes.append('extract_constant_to_initializer')
    # passes.append('eliminate_deadend')
    passes.append('eliminate_nop_transpose')
    passes.append('fuse_add_bias_into_conv')
    passes.append('fuse_bn_into_conv')
    passes.append('fuse_matmul_add_bias_into_gemm')
    passes.append('fuse_transpose_into_gemm')
    logger.info("passes: %s", passes)

    optimized_model = optimizer.optimize(original_model, passes)
    inferred_model = shape_inference.infer_shapes(optimized_model)
    onnx.checker.check_model(inferred_model)
    onnx.save(optimized_model, model_path)
    logger.info('optimize finish. save model to %s', model_path)


if __name__ == "__main__":

    if len(sys.argv) <2:
        print ("Usage:", sys.argv[0], "onnx_model  [output_model]")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = input_file.split(".onnx")[0] + "_optimized.onnx"
    if (len(sys.argv) == 3):
        output_file = sys.argv[2]


    # ---- step0. onnx simplifier.fix bug https://github.com/onnx/onnx/issues/2417
    onnx_sim = onnx_simplifier.simplify(sys.argv[1], check_n=1, perform_optimization=True, input_shapes={})
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
    logger.info('save onnx model %s ...', output_file)
    onnx.save(onnx_model, output_file)

    # ---- step3. check inference reslut
    # check model, inference compare
    logger.info("inference test...")
    input_shape = graph.input.dims
    input_data = np.random.randint(0,255, size=input_shape).astype(np.float32)
    logger.info("input_shape:%s", str(input_shape))

    model_1 = onnx.load(sys.argv[1])
    session_1 = backend.prepare(model_1,  strict=False)
    output_1 = session_1.run(input_data)
    logger.info("input model test finish")

    model_2 = onnx.load(output_file)
    session_2 = backend.prepare(model_2,  strict=False)
    output_2 = session_2.run(input_data)
    logger.info("output model test finish")

    # compare result
    np.testing.assert_allclose(output_1, output_2, rtol=1e-3    , atol=1e-4)
    logger.info("test ok")

    # ---- step4.check  operator support
    ret = onnx_check.ir_op_check(graph)
    if ret == False:
        logger.warn("onnx operator check not pass!")
    else:
        logger.info("check pass")

