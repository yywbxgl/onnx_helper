import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from onnx import helper, shape_inference
import numpy as np
import onnxruntime.backend as backend
import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
# fmt = "%(filename)s:%(lineno)d %(levelname)s - %(message)s"
# coloredlogs.install(level="DEBUG", fmt=fmt)
coloredlogs.install(level="INFO", fmt=fmt)
logger = logging.getLogger(__name__)

from IR import pb_to_ir
from IR import ir_to_pb
# from optimizer import operator_convert
from checker import onnx_check
from simplifyer import onnx_simplifier
from optimizer import optimize_graph

if __name__ == "__main__":

    if len(sys.argv) <2:
        print ("Usage:", sys.argv[0], "onnx_model  [output_model]")
        sys.exit(-1)

    input_file = sys.argv[1]
    output_file = input_file.split(".onnx")[0] + "_optimized.onnx"
    if (len(sys.argv) == 3):
        output_file = sys.argv[2]

    # ---- step0. simplify onnx model
    # onnx_sim = onnx_simplifier.simplify(input_file, input_shape=[1,224,224,3])
    # onnx_sim = onnx_simplifier.simplify(input_file, input_shape=[1,3,12,12])
    onnx_sim = onnx_simplifier.simplify(input_file)
    # onnx_sim = onnx_simplifier.change_version(onnx_sim)
    onnx.save(onnx_sim, output_file)
    logger.info('save simplified model: %s ...', output_file)


    # ---- step1. optimize ir graph
    # pb_to_ir
    # graph = pb_to_ir.convert(input_file)
    graph = pb_to_ir.convert(output_file)
    graph.dump()

    # ir graph optimize
    # 注意优化顺序！！！
    pass_list = [

        "eliminate_dropout",
        "eliminate_identity",
        "eliminate_pad",


        "convert_constant_to_init",  # 这五个顺序注意
        "convert_shape_to_init",  
        "convert_gather_to_init", 
        "convert_unsuqeeze_to_init",
        "convert_concat_to_init",
        "cast_to_init",
        "transpose_to_init",
        "convert_constantOfShape_to_init",

        "convert_flatten_to_reshape",
        "convert_reduceMean_to_globalAveragePool",
        "globbalMaxPool_to_maxPool",

        "fuse_pad_into_averagePool",
        "fuse_pad_into_maxPool",
        "fuse_pad_into_conv",

        "reshape_consecutive_eliminate",
        "reshape_nop_eliminate",

        # "transpose_input",            
        # "transpose_into_reshape",
        # "transpose_into_reducemean",
        # "transpose_into_reshape_prenode",
        # "transpose_eliminate",     # tranpose pass 慎用
        # "softmax_swap_down",
        # "transpose_swap_down",

        "squeeze_to_reshape",
        "unsqueeze_to_reshape",

        "leakRelu_to_PRelu",
        "matmul_to_gemm",

        "slice_nop_eliminate",
    ]

    # pass_list = ["softmax_swap_down", "reshape_consecutive_eliminate"]

    graph = optimize_graph(graph, pass_list)
    # graph.dump()
  
    # pb_to_ir
    onnx_ori = onnx.load(output_file)
    onnx_model = ir_to_pb.convert(graph)
    onnx.save(onnx_model, output_file)
    logger.info('save onnx model %s ...', output_file)


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
    onnx_model.producer_name =  "pytorch"
    file_name = output_file + "_ir3.onnx"
    loadable_name = output_file + "_ir3.nbdla"
    logger.info('save onnx model %s ...', file_name)
    onnx.save(onnx_model, file_name)
    onnx_simplifier.test_conveted_model(onnx_ori, file_name)

    complier = "../tools/ys13_bin_onnc.nv_large  "
    cmd = complier + file_name + "  -o " + loadable_name
    ret = os.system(cmd)
    if ret != 256:
        logger.info('compile success. save loadable %s ...', loadable_name)

