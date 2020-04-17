import onnx
import sys
import copy
from onnx import helper, shape_inference
# from onnx.tools import update_model_dims
from typing import List, Dict, Union, Optional, Tuple
from onnx import optimizer
import onnxruntime.backend as backend
import numpy as np

import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
# fmt = "%(filename)s:%(lineno)d %(levelname)s - %(message)s"
# coloredlogs.install(level="INFO", fmt=fmt)
# coloredlogs.install(level="DEBUG", fmt=fmt)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# 获取graph的input 
def get_input_names(model: onnx. 
ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names

#Due to a onnx bug, https://github.com/onnx/onnx/issues/2417, we need to add missing initializers into inputs
def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend(
                [onnx.ValueInfoProto(name=x.name,
                                     type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                           shape=shape)))])
    return model


def update_input_output_shape(model, input_shape):
    input_names = get_input_names(model)
    if len(input_names) != 1:
        print("not support multi input")
    for i in model.graph.input:
        if i.name == input_names[0]:
            for num, val in enumerate(i.type.tensor_type.shape.dim):
                val.dim_value = input_shape[num]
            logger.info("change input shape:%s", input_shape)
    return model


def convert_auto_pad(model):
    for node in model.graph.node:
        if node.op_type == "Conv" or node.op_type == "AveragePool" or node.op_type == "MaxPool":
            for i, attr in  enumerate(node.attribute):
                if attr.name == "auto_pad":
                    if attr.s == b'VALID':
                        logger.debug("convert_auto_pad, %s %s", node.name, attr.s)
                        new_attr = helper.make_attribute("pads",[0,0,0,0])
                        node.attribute.append(new_attr)
                        node.attribute.pop(i)
    return model


def test_conveted_model(model_ori, model_opt, input_shape=None):
    if type(model_ori) == str:
        model_ori = onnx.load(model_ori)

    if type(model_opt) == str:
        model_opt = onnx.load(model_opt)

    input_name = get_input_names(model_opt)[0]
    input_s = input_shape
    for i in model_opt.graph.input:
        if i.name == input_name:
            input_s = [t.dim_value for t in i.type.tensor_type.shape.dim]
    logger.info("test data: %s", input_s)
    input_data = np.random.randint(0,255, size=input_s).astype(np.float32)
    session_1 = backend.prepare(model_ori,  strict=False)
    output_1 = session_1.run(input_data)
    logger.debug("model_ori finish")

    session_2 = backend.prepare(model_opt,  strict=False)
    output_2 = session_2.run(input_data)
    logger.debug("model_opt finish")
    np.testing.assert_allclose(output_1, output_2, rtol=1e-3    , atol=1e-4)
    logger.info("test pass")


def delete_default_attr(model):
    for node in model.graph.node:
        # conv/pool   ceil_model=0 ; auto_pad= "NOTSET"
        if node.op_type == "MaxPool" or node.op_type == "AveragePool":
            for i, attr in  enumerate(node.attribute):
                if attr.name == "auto_pad":
                    if attr.s == b'NOTSET':
                        logger.info("delete attr auto_pad, %s %s", node.name, attr.s)
                        node.attribute.pop(i)
            for i, attr in  enumerate(node.attribute):
                if attr.name == "ceil_mode":
                    if attr.i == 0:
                        logger.info("delete attr ceil_mode, %s %s", node.name, attr.i)
                        node.attribute.pop(i)

        # default axis = 1 
        if node.op_type == "Softmax":
            for i, attr in  enumerate(node.attribute):
                if attr.name == "axis":
                    if attr.i == 1 or  attr.i == -1: # ps: 当input [n*c*h*w] 的n为1时，softmax的axis=-1 等价于axis=1 
                        logger.warn("delete softmax axis, %s %d", node.name, attr.i)
                        node.attribute.pop(i)
    return model


def change_version(model):
    logger.info("change version %d/%d  to 3/8", model.ir_version, model.opset_import[0].version)
    model.ir_version = 3
    model.opset_import[0].version = 8
    return model


    
def simplify(model, input_shape=None):
    onnx_model = onnx.load(model)
    model_ori = copy.deepcopy(onnx_model)

    # fix bug
    onnx_model = add_initializers_into_inputs(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    
    # change dynamic input 
    if input_shape != None:
        onnx_model = update_input_output_shape(onnx_model, input_shape)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(onnx_model)

    # optimize
    passes = ["eliminate_deadend", 
    # "eliminate_identity", 
    # "eliminate_nop_dropout",
    # "eliminate_nop_monotone_argmax",
    # "eliminate_nop_pad",
    # "eliminate_nop_transpose",
    # "eliminate_unused_initializer",
    # "extract_constant_to_initializer",
    
    # "fuse_consecutive_concats",
    # "fuse_add_bias_into_conv",
    # "fuse_bn_into_conv",
    "fuse_matmul_add_bias_into_gemm",
    # "fuse_pad_into_conv",
    # "fuse_transpose_into_gemm",
    ]
    onnx_model = optimizer.optimize(onnx_model, passes)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    # convert_auto_pad
    onnx_model = convert_auto_pad(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    # delete_default_attr
    onnx_model = delete_default_attr(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    # onnx_model = change_version(onnx_model)

    # test converted model
    test_conveted_model(model_ori, onnx_model, input_shape)

    return onnx_model



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnxModel  outputName")
        sys.exit(-1)

    model = simplify(sys.argv[1], input_shape=[1,224,224,3])
    onnx.save(model, sys.argv[2])