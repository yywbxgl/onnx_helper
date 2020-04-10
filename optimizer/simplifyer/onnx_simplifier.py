import onnx
import sys
import copy
from onnx import helper, shape_inference
# from onnx.tools import update_model_dims
from typing import List, Dict, Union, Optional, Tuple
from onnx import optimizer

# 获取graph的input 
def get_input_names(model: onnx.ModelProto) -> List[str]:
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
            print("change input shape:", i.type.tensor_type.shape.dim)
    return model

    
    
def simplify(model, input_shape=None):
    onnx_model = onnx.load(model)
    model_ori = copy.deepcopy(onnx_model)

    # fix bug
    onnx_model = add_initializers_into_inputs(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    
    # change dynamic input 
    onnx_model = update_input_output_shape(onnx_model, input_shape)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    # optimize
    passes = ["eliminate_deadend", 
    "eliminate_identity", 
    "eliminate_nop_dropout",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_transpose",
    "eliminate_unused_initializer",
    "extract_constant_to_initializer",
    "fuse_consecutive_concats",
    "fuse_add_bias_into_conv",
    "fuse_bn_into_conv",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
    "fuse_transpose_into_gemm",
    ]
    onnx_model = optimizer.optimize(onnx_model, passes)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    return onnx_model




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnxModel  outputName")
        sys.exit(-1)

    model = simplify(sys.argv[1])
    onnx.save(model, sys.argv[2])