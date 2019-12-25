import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from onnx import shape_inference

from IR import ir

# def search_node_by_input(input_name, graphProto):
#     node_list = []
#     print("search_next_node by name:", input_name)
#     for node in graphProto.node:
#         if input_name in node.input:
#             node_list.append(node)
#             print("find node ", node.name)
#     return node_list


def protoTensor_to_irValue(proto_tensor):
    ir_value = ir.Value()
    ir_value.name = proto_tensor.name
    ir_value.data_type = proto_tensor.data_type
    ir_value.dims = [n for n in proto_tensor.dims]
    if len(ir_value.dims)  == 0:
        # ir_value.dims = [0]  # 0维度，表示标量
        print(ir_value.name, "is dims 0")
        
    ir_value.data += proto_tensor.float_data
    ir_value.data += proto_tensor.int32_data
    ir_value.data += proto_tensor.string_data
    ir_value.data += proto_tensor.int64_data
    ir_value.data += proto_tensor.double_data
    ir_value.data += proto_tensor.uint64_data
    if (len(proto_tensor.external_data) != 0):
        print("can not parse external_data.")

    if (len(proto_tensor.raw_data) != 0):
        print("weight %s is raw data."%(ir_value.name))
        ir_value.data += proto_tensor.raw_data
        ir_value.raw = True
        #todo  把raw_data根据data_type转为具体的数值, 新增接口实现
        # print(ir_value.name, "get raw data")

    return ir_value


def protoValueInfo_to_irValue(proto_value_info):
    feature_map = ir.Value()
    feature_map.name = proto_value_info.name
    feature_map.data_type = proto_value_info.type.tensor_type.elem_type
    for i in proto_value_info.type.tensor_type.shape.dim:
        feature_map.dims.append(i.dim_value)
    return feature_map


def protoAttribute_to_irValue(proto_attribute):
    attr = ir.Value()
    attr.name = proto_attribute.name
    # 注意pb中 AttributeType的type 与 ValueInfo中的type 定义不同
    attr.data_type = proto_attribute.type 
    if proto_attribute.type  <= 5:
        attr.dims.append(1)
        if proto_attribute.type == 1: 
            attr.data.append(proto_attribute.f)
        if proto_attribute.type == 2: 
            attr.data.append(proto_attribute.i)
        if proto_attribute.type == 3:  
            attr.data.append(proto_attribute.s)
        if proto_attribute.type == 4: 
            attr.data.append(proto_attribute.t)   # todo parse
        if proto_attribute.type == 5:  
            attr.data.append(proto_attribute.g)   # todo parse
        # print(attr.name, len(attr.data), attr.data, attr.dims, attr.data_type)
    else :
        total_num = len(proto_attribute.floats) + len(proto_attribute.ints) + \
            len(proto_attribute.strings) + len(proto_attribute.tensors) + len(proto_attribute.graphs)
        attr.dims.append(total_num)
        attr.data += (proto_attribute.ints)
        attr.data += (proto_attribute.floats)
        attr.data += (proto_attribute.strings)
        attr.data += (proto_attribute.tensors)  # todo parse
        attr.data += (proto_attribute.graphs)   # todo parse
        # print("---", attr.name, len(attr.data), attr.data,  attr.dims, attr.data_type)
    return attr


def convert(onnx_model_file):

    print("convert pb to ir. load model ...")
    ModelProto = onnx.load(onnx_model_file)
    print("check onnx model ...")
    onnx.checker.check_model(ModelProto)
    print("shape inference onnx model ...")
    ModelProto = shape_inference.infer_shapes(ModelProto)

    # ----- 解析 model_proto层-----------
    print("---- model info ----")
    print("ir_version:", ModelProto.ir_version)
    print("opsert_import:", ModelProto.opset_import[0].domain, ModelProto.opset_import[0].version)
    print("producer_name:", ModelProto.producer_name)

    proto_graph = ModelProto.graph
    ir_graph = ir.Graph()
    if (proto_graph.name != ""):
        ir_graph.name = proto_graph.name 
        print("graph name:", ir_graph.name)

    # ----- 添加ir_input, ir_output -------
    init_list = [i.name for i in proto_graph.initializer]
    input_list = [i.name for i in proto_graph.input]
    if len(input_list) - len(init_list) > 1:
        print("!!! can not parse multi input.")
        sys.exit(-1)
    for i in proto_graph.input:
        if i.name not in init_list:
            ir_graph.input = protoValueInfo_to_irValue(i)    
    print("input: ", ir_graph.input.name , str(ir_graph.input.dims))

    for dim in ir_graph.input.dims:
        if dim == 0:
            print("!!! can not parse dynamic input.")
            sys.exit(-1)

    if len(proto_graph.output) > 1:
        print("!!! can not parse multi output.")
        sys.exit(-1)
    ir_graph.output = protoValueInfo_to_irValue(proto_graph.output[0])
    print("output:", ir_graph.output.name,  ir_graph.output.dims)

    # ----- 添加 init_list -----
    print("----------------------")
    init_dict = {}
    for proto_init in proto_graph.initializer:
        ir_value = protoTensor_to_irValue(proto_init)
        ir_value.init = True
        init_dict[ir_value.name] = ir_value

    for i in init_dict:
        print("init-weight: ", init_dict[i].name, init_dict[i].dims)
        

    # ---- 添加mid feature list -----
    print("----------------------")
    mid_feature_dict = {}
    for proto_node in proto_graph.node:
        for output in proto_node.output:
            if output != ir_graph.output.name:
                feature_map = ir.Value()
                feature_map.name = output
                mid_feature_dict[feature_map.name] = feature_map
    for value_info in proto_graph.value_info:
        mid_feature_dict[value_info.name].data_type = value_info.type.tensor_type.elem_type
        for i in value_info.type.tensor_type.shape.dim:
            mid_feature_dict[value_info.name].dims.append(i.dim_value)

    for i in mid_feature_dict:
        print("mid-feature: ", mid_feature_dict[i].name, mid_feature_dict[i].dims)

    # ----- 添加ir_node ------
    print("----------------------")
    for proto_node in proto_graph.node:
        ir_node = ir.Node()
        ir_node.name = proto_node.name
        ir_node.op_type = proto_node.op_type
        for attr in proto_node.attribute:
            ir_node.attribute.append(protoAttribute_to_irValue(attr))
        for i in proto_node.input:
            if i in init_dict:
                ir_node.weight.append(init_dict[i])
            elif i in mid_feature_dict:
                ir_node.input.append(mid_feature_dict[i])
            elif i == ir_graph.input.name:
                ir_node.input.append(ir_graph.input)
            else:
                print("input [%s] not find"%(i))
            
        for i in proto_node.output:
            if i in mid_feature_dict:
                ir_node.output.append(mid_feature_dict[i])
            elif i == ir_graph.output.name:
                ir_node.output.append(ir_graph.output)
            else:
                print("!!! output [%s] not find"%(i))

        ir_graph.node_list.append(ir_node)


    for i in ir_graph.node_list:
        print("node = ", i.name) 
   
    updata_graph(ir_graph)

    print("convert to ir graph success !")
    return ir_graph


def updata_graph(ir_graph):
    # ----- 遍历graph，填充pre_node 与 next_node ------
    for node in ir_graph.node_list:
        # print("node[%s] to find pre_node"%(node.name))
        for i in node.input:
            for node2 in ir_graph.node_list:
                if i in node2.output:
                    node.pre_node.append(node2)
                    # print(node2.name)

        # print("node[%s] to find next_node"%(node.name))
        for o in node.output:
            for node2 in ir_graph.node_list:
                if o in node2.input:
                    node.next_node.append(node2)
                    # print(node2.name)

    # ----- 重命名Node.name, 确保名称唯一------
    print("---------------------")
    print("rename node name ...")
    for index, node in  enumerate(ir_graph.node_list):
        node.name = node.op_type + "_" + str(index)


def dump(ir_graph):
    print("-------- ir_grapg dump --------------")
    print("graph_input  =", ir_graph.input.name,  ir_graph.input.dims)
    print("graph_output =", ir_graph.output.name,  ir_graph.output.dims)
    for i in ir_graph.node_list:
        print("node = ", i.name) 
        for inp in i.input:
            print("\tinput:", inp.name, inp.dims)
        for out in i.output:
            print("\toutput:", out.name, out.dims)
        for attr in i.attribute:
            print("\tattr:", attr.name, attr.data)
        for w in i.weight:
            print("\tweight:", w.name, w.dims)
    print("-----------------------------------")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " OnnxModel")
        sys.exit(-1)

    graph = convert(sys.argv[1])
    dump(graph)
    