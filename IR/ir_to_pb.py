import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
import onnx

from IR import ir 

def irValue_to_protoValueInfo(ir_value):
    out = helper.make_tensor_value_info(
        name = ir_value.name,
        elem_type = ir_value.data_type,  
        shape = ir_value.dims
    )
    return out


def irValue_to_protoTensor(ir_value):

    if (ir_value.raw == True):
        out = helper.make_tensor(
            name = ir_value.name,
            data_type = ir_value.data_type,
            dims = ir_value.dims,
            vals = bytes(ir_value.data),
            raw = True 
        )   
    else:
        out = helper.make_tensor(
            name = ir_value.name,
            data_type = ir_value.data_type,
            dims = ir_value.dims,
            vals = ir_value.data,
        )

    return out


def convert(ir_graph):

    print("convert ir to pb ...")

    # ------ make input and output  -----------
    output_data = irValue_to_protoValueInfo(ir_graph.output)
    inputs = [irValue_to_protoValueInfo(ir_graph.input)]
    for node in ir_graph.node_list:
        for i in node.weight:
            temp = irValue_to_protoValueInfo(i)
            inputs.append(temp)
    
    print("inputs: \n", [i.name for i in inputs])
    print("outputs: \n", output_data.name)

    # ------ make initializers -----------
    initializers = []
    for node in ir_graph.node_list:
        for w in node.weight:
            init = irValue_to_protoTensor(w)
            initializers.append(init)
    print("initializers: \n", [i.name for i in initializers])
    

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

        for attr in node.attribute:
            if attr.data_type <= 5:  # fixed
                temp = onnx.helper.make_attribute(attr.name, attr.data[0])
            else:
                temp = onnx.helper.make_attribute(attr.name, attr.data)
            # print("attr:", temp.name, temp.type)
            proto_node.attribute.append(temp)
        nodes.append(proto_node)

    print("node: \n", [i.name for i in nodes])


    # ------ make graph -----------
    if ir_graph.name == "":
        ir_graph.name = "nbdla-test"
    graph_def = helper.make_graph(
        name = ir_graph.name,
        nodes = nodes,
        inputs = inputs,
        outputs = [output_data],
        initializer = initializers,
    )
    # print(graph_def)

    # print("-------------------------")
    # print(onnx.helper.printable_graph(graph_def))
    # print("-------------------------")

    # Create the model (ModelProto)
    onnx_model = helper.make_model(graph_def, producer_name='nbdla')

    print('check onnx model ...')
    onnx.checker.check_model(onnx_model)
    print('shepe inference onnx model ...')
    onnx_model = shape_inference.infer_shapes(onnx_model)

    print("convert ir to pb success.")
    # print('save onnx model ...')
    # onnx.save(onnx_model, "test.onnx")

    return onnx_model