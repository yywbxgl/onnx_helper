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
# logger.setLevel(logging.DEBUG)


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

    if input_shape != None:
        for i in model.graph.input:
            if i.name == input_names[0]:
                for num, val in enumerate(i.type.tensor_type.shape.dim):
                    val.dim_value = input_shape[num]
                logger.warn("change input shape:%s", input_shape)
    
    # 修改input  0->1
    for t in model.graph.input:
        for dims in  t.type.tensor_type.shape.dim:
            if dims.dim_value == 0:
                logger.warn("change input shape")
                dims.dim_value = 1

	# 修改output  0->1
    # for t in model.graph.output:
    #     for dims in  t.type.tensor_type.shape.dim:
    #         if dims.dim_value == 0:
    #             logger.warn("change output shape")
    #             dims.dim_value = 1

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
                    elif attr.s == b"SAME_UPPER" or attr.s == b"SAME_LOWER":
                        kernal_shape = None
                        stride = None
                        for attr2 in node.attribute:
                            if attr2.name == "kernel_shape":
                                kernal_shape = attr2.ints 
                            elif attr2.name == "strides":
                                stride = attr2.ints    
                        logger.info("convert_auto_pad: %s  %s,  kernal_shape: %s, stride:%s", node.name, attr.s, kernal_shape, stride)
                        
                        if kernal_shape == None or stride == None:
                            logger.warn("can not find kernal_shape and stride.")
                            break

                        # # output_shape = ceil (input_size/stride)  当stride ==1 时 shape形状不变
                        # total_pad_h = kernal_shape[0] - stride[0]
                        # total_pad_h_left = total_pad_h_right = total_pad_h//2
                        # total_pad_w = kernal_shape[1] - stride[1]
                        # total_pad_w_left = total_pad_w_right = total_pad_w//2

                        input_shape = []
                        output_shape = []
                        # logger.warn([o.name for o in model.graph.value_info])
                        for o in model.graph.value_info:
                            if o.name == node.input[0]:
                                input_shape = [t.dim_value for t in o.type.tensor_type.shape.dim]
                                logger.info("get input shape %s", input_shape)
                            if o.name == node.output[0]:
                                output_shape = [t.dim_value for t in o.type.tensor_type.shape.dim]
                                logger.info("get output shape %s", output_shape)

                        # conv 未网络最后一层时， shape 从 graph.output 取
                        if len(output_shape) == 0:
                            for o in model.graph.output:
                                if o.name == node.output[0]:
                                    output_shape = [t.dim_value for t in o.type.tensor_type.shape.dim]
                                    logger.info("get output shape %s", output_shape)
                        
                        if len(input_shape) == 0:
                            logger.warn("can not get input shape.")
                            break

                        input_h = input_shape[2]
                        input_w = input_shape[3]
                        output_h = output_shape[2]
                        output_w = output_shape[3]

                        # output_shape = ceil(input_size/stride)
                        # stride = 1 时  pad = kernal_shape - stride
                        pad_h = (output_h -1)*stride[0] + kernal_shape[0] - input_h
                        pad_w = (output_w -1)*stride[1] + kernal_shape[1] - input_w
                        
                        if pad_h < 0 or  pad_w < 0 :
                            pad_h = pad_h + stride[0] -1
                            pad_w = pad_w + stride[1] -1
                            logger.warn("convert pad value. %s  %s", pad_h, pad_w)
                            if pad_h < 0 or  pad_w < 0 :
                                logger.warn("can not support %s  %s", pad_h, pad_w)
                                break

                        pad_h_left = pad_h_right = pad_h//2
                        pad_w_left = pad_w_right = pad_w//2
                        if pad_h % 2 == 1:
                            if attr.s == b"SAME_LOWER":
                                pad_h_left = pad_h_left + 1
                            else:
                                pad_h_right = pad_h_right + 1
                        if pad_w % 2 == 1:
                            if attr.s == b"SAME_LOWER":
                                pad_w_left = pad_w_left + 1
                            else:
                                pad_w_right = pad_w_right + 1
                        

                        # 添加pads属性  删除auto_pad
                        new_attr = helper.make_attribute("pads",[pad_h_left,pad_w_left,pad_h_right,pad_w_right])
                        if stride[0] != 1 or stride[1] != 1:
                            logger.warn("stride= %s, convert to pads %s ",stride, [pad_h_left,pad_w_left,pad_h_right,pad_w_right])
                        else:
                            logger.info("convert to pads %s ", [pad_h_left,pad_w_left,pad_h_right,pad_w_right])
                        node.attribute.append(new_attr)
                        node.attribute.pop(i)

    return model


def test_conveted_model(model_ori, model_opt):
    if type(model_ori) == str:
        model_ori = onnx.load(model_ori)

    if type(model_opt) == str:
        model_opt = onnx.load(model_opt)


    input_name = get_input_names(model_ori)[0]
    input_shape_ori = []
    for i in model_ori.graph.input:
        if i.name == input_name:
            input_shape_ori = [t.dim_value for t in i.type.tensor_type.shape.dim]

    input_name = get_input_names(model_opt)[0]
    input_shape_opt = []
    for i in model_opt.graph.input:
        if i.name == input_name:
            input_shape_opt = [t.dim_value for t in i.type.tensor_type.shape.dim]

    # 如果ori model的为动态输入  那么使用opt model的输入
    if 0 in input_shape_ori:
        input_shape_ori = input_shape_opt

    # input_data = np.random.randint(0,255, size=input_shape_ori).astype(np.float32).astype(np.uint8)
    input_data = np.random.randint(0,255, size=input_shape_ori).astype(np.float32)
    logger.info("test model_ori %s", input_data.shape)
    session_1 = backend.prepare(model_ori,  strict=False)
    output_1 = session_1.run(input_data)
    logger.debug("model_ori finish")

    # transpose input shape
    [n,c,h,w] = input_shape_opt
    if input_data.shape == (n,h,w,c):
        logger.info("transpose input shape  %s to %s ", list(input_data.shape), input_shape_opt)
        input_data = np.transpose(input_data, [0,3,1,2])

    logger.info("test model_opt %s",  input_data.shape)
    session_2 = backend.prepare(model_opt,  strict=False)
    output_2 = session_2.run(input_data)
    logger.debug("model_opt finish")

    if (len(output_1) != len(output_2)):
        logger.error("output shape not equal.")

    # 有多个output时候， 分别对比output
    for i in range(len(output_1)):
        np.testing.assert_allclose(output_1[i], output_2[i], rtol=1e-3, atol=1e-4)
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


# 删除指定名称的node 只能删除单输入输出的node
def eliminate_one_node(model, name):
    if type(model) == str:
        model = onnx.load(model)

    for node in model.graph.node:
        if node.name == name:
            logger.warn("----eliminate node %s", node.name)
            n_input = node.input[0]
            n_output = node.output[0]
            # logger.warn("%s  ,%s", n_input,n_output)

            # 如果是最后一个node, 修改上个node的output
            if n_output == model.graph.output[0].name:
                for node2 in model.graph.node:
                    if node2.output[0] == n_input:
                        logger.warn("---- find prev node: %s", node2.name)
                        node2.output[0] = n_output
            else:
            # 修改下一个node的input
                for node2 in model.graph.node:
                    if len(node2.input) != 0 and node2.input[0] == n_output:
                        logger.warn("---- find next node: %s", node2.name)
                        node2.input[0] = n_input

            # 删除value info
            for value in model.graph.value_info:
                if value.name == n_input:
                    logger.warn("----eliminate value_info %s",value.name)
                    model.graph.value_info.remove(value)

            # 删除当前node  
            model.graph.node.remove(node)
            # return model
    

    return model


def concat_output(model):
    if type(model) == str:
        model = onnx.load(model)

    # add concat node
    new_node = onnx.NodeProto()
    new_node.name = "concat_outputs"
    new_node.op_type = "Concat"
    new_attribute = onnx.AttributeProto()
    new_attribute.name = "axis"
    new_attribute.type = 2
    new_attribute.i = 1
    new_node.attribute.append(new_attribute)
    new_node.output.append("concat_outputs_ret")
    for i in model.graph.output:
        new_node.input.append(i.name)
    model.graph.node.append(new_node)

    print("------------")

    # modify grapg input
    # for i in  model.graph.output:
    #     model.graph.input.append(i)

    # modify grapg output
    while(len( model.graph.output) > 1):
        model.graph.output.pop()
    model.graph.output[0].name = "concat_outputs_ret"
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 15

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
    passes = [
    "eliminate_deadend", 
    "eliminate_identity", 
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

    onnx_model = change_version(onnx_model)

    # test converted model
    test_conveted_model(model_ori, onnx_model)

    return onnx_model



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnxModel  outputName")
        sys.exit(-1)

    # onnx_model = simplify(sys.argv[1])
    # onnx_model = eliminate_one_node(sys.argv[1], "Sigmoid_12")
    onnx_model = concat_output(sys.argv[1])
    
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, sys.argv[2])