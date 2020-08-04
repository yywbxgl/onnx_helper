import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
import onnx
import logging
import numpy as np
logger = logging.getLogger(__name__)

from IR import ir 
from IR import pb_to_ir

def irValue_to_protoValueInfo(ir_value):
    out = helper.make_tensor_value_info(
        name = ir_value.name,
        elem_type = ir_value.data_type,  
        shape = ir_value.dims
    )
    return out


def irValue_to_protoTensor(ir_value):

    if ir_value.raw == True:
        out = helper.make_tensor(
            name = ir_value.name,
            data_type = ir_value.data_type,
            dims = ir_value.dims,
            vals = bytes(ir_value.data),
            raw = True 
        )   
    else:
        data = np.array(ir_value.data).flatten()
        out = helper.make_tensor(
            name = ir_value.name,
            data_type = ir_value.data_type,
            dims = ir_value.dims,
            vals = data,
        )

    return out


def convert(ir_graph):

    logger.info("convert ir to pb ...")

    # ------ make input and output  -----------
    inputs = []
    for i in ir_graph.input:
        temp = irValue_to_protoValueInfo(i)
        inputs.append(temp)
        logger.debug("inputs: %s", i.name)

    for node in ir_graph.node_list:
        for i in node.weight:
            if i.name not in [t.name for t in inputs]:
                temp = irValue_to_protoValueInfo(i)
                inputs.append(temp)
        for i in node.input:  # add initializers to input
            if i.init == True:
                if i.name not in [t.name for t in inputs]:
                    temp = irValue_to_protoValueInfo (i)
                    inputs.append(temp)
    logger.debug("inputs: %s", [i.name for i in inputs])

    output_data = []
    for i in ir_graph.output:
        temp = irValue_to_protoValueInfo(i)
        output_data.append(temp)
        logger.debug("outputs: %s", i.name)

    # ------ make initializers -----------
    initializers = []
    for node in ir_graph.node_list:
        for w in node.weight:
            if w.name not in [t.name for t in initializers]:
                init = irValue_to_protoTensor(w)
                initializers.append(init)
        for i in node.input:
            if i.init == True:
                if i.name not in [t.name for t in initializers]:
                    temp = irValue_to_protoTensor(i)
                    initializers.append(temp)
                    logger.info("add init input %s", i.name)
    logger.debug("initializers: %s", [i.name for i in initializers])
    

    # ------ make node -----------
    nodes = []
    for node in ir_graph.node_list:
        inp = [i.name for i in node.input]
        inp += [i.name for i in node.weight]
        out = [i.name for i in node.output]
        proto_node = onnx.helper.make_node(
            name =node.name,
            op_type = node.op_type,
            inputs = inp,
            outputs = out,
        )
        # print("create node. op_type=",node.op_type ,"input:", inp, "output:", out)

        for attr in node.attribute:
            if attr.data_type <= 5:  # fixed
                temp = onnx.helper.make_attribute(attr.name, attr.data[0])
            else:
                temp = onnx.helper.make_attribute(attr.name, attr.data)
            # print("attr:", temp.name, temp.type)
            proto_node.attribute.append(temp)
        nodes.append(proto_node)

    logger.debug("node: %s", [i.name for i in nodes])


    # ------ make graph -----------
    if ir_graph.name == "":
        ir_graph.name = "nbdla-test"
    graph_def = helper.make_graph(
        name = ir_graph.name,
        nodes = nodes,
        inputs = inputs,
        outputs = output_data,
        initializer = initializers,
    )
    # print(graph_def)

    # Create the model (ModelProto)
    if ir_graph.opset !=0 and ir_graph.ir_version !=0:
        opset = helper.make_opsetid("", ir_graph.opset)
        onnx_model = helper.make_model(graph_def, producer_name='YKX-converter', 
            opset_imports= [opset], ir_version=ir_graph.ir_version)
    else:
        onnx_model = helper.make_model(graph_def, producer_name='YKX-converter')

    logger.info('check onnx model ...')
    onnx.checker.check_model(onnx_model)
    logger.info('shepe inference onnx model ...')
    onnx_model = shape_inference.infer_shapes(onnx_model)

    logger.info("convert ir to pb success.")
    # logger.info('save onnx model ...')
    # onnx.save(onnx_model, "test.onnx")
    

    return onnx_model