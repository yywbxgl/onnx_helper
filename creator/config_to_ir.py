import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import collections
import numpy as np

from IR import ir
from creator import ir_to_config

def parse_to_ir_value(line):
    out = ir.Value()
    temp = line.split(":", 1)[1].strip()  # 用 ":" 分割
    contents = temp.split(";")
    name = contents[0].strip()  # 用","分割一次，第一个值为name
    dims = contents[1].strip()  # 用","分割一次，第二个值为dims
    out.name = name
    out.dims = eval(dims)
    out.data_type = 1 # todo  输入只支持float 类型
    return out


def parse_weight_to_value(line):
    out = ir.Value()
    content = line.split(":", 1)[1].strip()  # 用 ":" 分割
    contents = content.split(";")
    name = contents[0].strip()  # 用";"分割一次，第一个值为name
    dims = contents[1].strip()  # 用";"分割一次，第二个值为dims

    out.name = name
    out.dims = eval(dims)

    total_size = 1
    for i in out.dims:
        total_size *= i
    # out.data = list(np.random.rand(total_size))
    out.data = np.random.rand(total_size)
    
    if (len(contents) > 2):
        type_str = contents[2].strip()
        out.data_type = ir.DataType[type_str].value
        print("%s set data type %s  %d"%(out.name, type_str, out.data_type))
        out.data = out.data.astype(np.int64)   # todo  reshape weight 参数强制改为int64
    else:
        out.data_type = 1 #默认为float 类型 

    return out


def parse_attr_to_value(line):
    out = ir.Value()

    temp = line.split(":", 1)[1].strip() # 用 ":" 分割
    name = temp.split(";", 1)[0].strip()   # 用","分割一次，第一个值为name
    value = temp.split(";", 1)[1].strip()  # 用","分割一次，第二个值为attribute的值
    out.name = name
    out.data = eval(value)
    out.dims = len(out.data)
    if out.dims == 1:
        if type(out.data[0]) == type(1.0):
            out.data_type = 1
        elif type(out.data[0]) == type(1):
            out.data_type = 2
        elif type(out.data[0]) == type(b'STRING'):
            out.data_type = 3
        else:
            print("!!! config error. ", line)
    else:
        if type(out.data[0]) == type(1.0):
            out.data_type = 6
        elif type(out.data[0]) == type(1):
            out.data_type = 7
        elif type(out.data[0]) == type(b'STRING'):
            out.data_type = 8
        else:
            print("!!! config error. ", line)
    return out


def parse_to_ir_node(lines):
    out_node = ir.Node()
    for line in lines:
        key = line.split(":", 1)[0].strip()
        value = line.split(":", 1)[1].strip()
        if "name" in key:
            out_node.name = value
        elif "type" in key:
            out_node.op_type = value
        elif "inputs" in key:
            for i in value.split(";"):
                temp = ir.Value()
                temp.name = i.strip()
                out_node.input.append(temp)
        elif "outputs" in key:
            for i in value.split(";"):
                temp = ir.Value()
                temp.name = i.strip()
                out_node.output.append(temp)
        elif "weight" in key:
            temp = parse_weight_to_value(line)
            out_node.weight.append(temp)
        elif "attr" in key:
            temp = parse_attr_to_value(line)
            out_node.attribute.append(temp)
        else:
            print("!!! config error.", line)


    return out_node


# 删除空行和 “}” 行
def remove_invaild_line(in_list):
    while "" in in_list:
        in_list.remove("")
    for i in in_list:
        if "}" in i:
            in_list.remove(i)
    return in_list


def importConfig(confg_file):
    out_graph = ir.Graph()

    # read config file
    print("load config file ...")
    f = open(confg_file, 'r')
    content = f.read().split("{")

    # parse input and output
    print("convert config to ir ...")
    lines = content[0].split("\n")
    remove_invaild_line(lines)
    for line in lines:
        key = line.split(":",1)[0].strip()
        if "input" in key:
            out_graph.input = parse_to_ir_value(line)
        elif "output" in key:
            out_graph.output = parse_to_ir_value(line)
        elif "graph" in key:
            out_graph.name = line.split(":",1)[1].strip()
        else:
            print("config error.", line)
    content.pop(0)

    # parse node
    for node_str in content:
        lines = node_str.split("\n")
        remove_invaild_line(lines)
        # print(lines)
        out_graph.node_list.append(parse_to_ir_node(lines))


    # ----- 遍历graph，填充pre_node 与 next_node ------
    for node in out_graph.node_list:
        # print("node[%s] to find pre_node"%(node.name))
        for i in node.input:
            for node2 in out_graph.node_list:
                if i in node2.output:
                    node.pre_node.append(node2)
                    # print(node2.name)

        # print("node[%s] to find next_node"%(node.name))
        for o in node.output:
            for node2 in out_graph.node_list:
                if o in node2.input:
                    node.next_node.append(node2)
                    # print(node2.name)
    
    # dump ir config
    print("convert config to ir success")
    ir_to_config.exportConfig(out_graph)
    print("config to ir success.")

    return out_graph


if __name__ == "__main__":

    from IR import pb_to_ir
    from IR import ir_to_pb
    import onnx

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " config_file")
        sys.exit(-1)

    # convrt config file to ir_graph
    test_graph = importConfig(sys.argv[1])

    # convrt ir_graph to onnx model
    model2 = ir_to_pb.convert(test_graph)

    # save onnx model
    onnx.save(model2, "test.onnx")


