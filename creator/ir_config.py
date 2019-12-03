import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import collections
import numpy as np

from IR import ir

def exportConfig(ir_graph):
    print("----------------------")
    config = "graph : " + ir_graph.name + "\n"
    config += "input : " + ir_graph.input.name + " , " + str(ir_graph.input.dims) + "\n"
    config += "output : " + ir_graph.output.name + " , " + str(ir_graph.output.dims) + "\n"

    for i in ir_graph.node_list:
        config += "{\n"
        config += "\tname: " + i.name + "\n"
        config += "\ttype: " + i.op_type + "\n"

        for n in range(len(i.input)):
            if n == 0:
                config += "\tinputs: " + i.input[n].name
            else:
                config += " , " + i.input[n].name
        # 每个node不一定有input
        if (len(i.input) != 0):
            config += "\n"

        for n in range(len(i.output)):
            if n == 0:
                config += "\toutputs: " + i.output[n].name
            else:
                config += " , " + i.output[n].name
        # 每个node必须指定output
        config += "\n"

        for w in i.weight:
            config += "\tweight: " + w.name + " , " + str(w.dims) + "\n"

        for attr in i.attribute:
            # todo parse tensors and graphs
            if attr.data_type in [5, 9, 10]:
                print("!!! can not parse attr. data type=", attr.data_type)
                sys.exit(-1)
            config += "\tattr: " + attr.name + " , " + str(attr.data) + "\n"

        config += "}\n"

    print(config)
    print("----------------------")

    return config


def exportConfig_json(ir_graph):
    config = collections.OrderedDict()
    config["input"] = collections.OrderedDict({"name":ir_graph.input.name, "dims":ir_graph.input.dims})
    config["output"] = collections.OrderedDict({"name":ir_graph.output.name, "dims":ir_graph.output.dims})
    config["node"] = []

    for node in ir_graph.node_list:
        temp=collections.OrderedDict()
        temp["name"] = node.name
        temp["op_type"] = node.op_type
        temp["inputs"] = [i.name for i in node.input]
        temp["outputs"] = [i.name for i in node.output]

        if len(node.attribute) != 0:
            temp["attribute"] = []
            for attr in node.attribute:
                # todo  attr vlaue 值为tensor或者graph时
                if attr.data_type in [4, 5, 9, 10]:
                    print("can not parse attr data", attr.name, attr.data_type)
                    sys.exit(-1)
                if attr.data_type == 3: # todo
                    data_str = bytes.decode(attr.data[0], encoding="utf-8")
                    temp["attribute"].append({"name":attr.name, "value":data_str})
                else:
                    temp["attribute"].append({"name":attr.name, "value":attr.data})
        
        if len(node.weight) != 0:
            temp["weights"] = []
            for w in node.weight:
                temp["weights"].append(collections.OrderedDict({"name":w.name, "dims":w.dims}))

        config["node"].append(temp)

    config_json = json.dumps(config, indent=4)
    print(config_json)

    return config_json


def save_config(config, file_name):
    f = open(file_name, "w")
    f.write(config)
    f.close()


def parse_to_ir_value(line):
    out = ir.Value()
    temp = line.split(":", 1)[1].strip()  # 用 ":" 分割
    name = temp.split(",", 1)[0].strip()  # 用","分割一次，第一个值为name
    dims = temp.split(",", 1)[1].strip()  # 用","分割一次，第二个值为dims
    out.name = name
    out.dims = eval(dims)
    out.data_type = 1 # todo inference 输入类型写死为 float 类型
    return out


def parse_weight_to_value(line):
    out = ir.Value()
    temp = line.split(":", 1)[1].strip() # 用 ":" 分割
    name = temp.split(",", 1)[0].strip() # 用","分割一次，第一个值为name
    dims = temp.split(",", 1)[1].strip()  # 用","分割一次，第二个值为dims或者npy文件
    out.name = name
    if ".npy" in dims:
        print("init weight by numpy", dims)
        np_data = np.load(dims)
        out.dims = list(np_data.shape)
        out.data_type = 1 # todo inference 输入类型写死为 float 类型
        out.data = list(np_data.flatten())
    else:
        out.dims = eval(dims)
        out.data_type = 1 # todo inference 输入类型写死为 float 类型 
        total_size = 1
        for i in out.dims:
            total_size *= i
        out.data = list(np.random.rand(total_size)) 

    return out


def parse_attr_to_value(line):
    out = ir.Value()

    temp = line.split(":", 1)[1].strip() # 用 ":" 分割
    name = temp.split(",", 1)[0].strip()   # 用","分割一次，第一个值为name
    value = temp.split(",", 1)[1].strip()  # 用","分割一次，第二个值为attribute的值
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


def parse_output_to_value(line):
    out = ir.Value()
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
            for i in value.split(","):
                temp = ir.Value()
                temp.name = i.strip()
                out_node.input.append(temp)
        elif "outputs" in key:
            for i in value.split(","):
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
    f = open("test.cfg", 'r')
    # f = open(confg_file, 'r')
    content = f.read().split("{")

    # parse input and output
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
    
    exportConfig(out_graph)
    return out_graph


if __name__ == "__main__":

    from IR import pb_to_ir
    from IR import ir_to_pb
    import onnx

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " OnnxModel")
        sys.exit(-1)

    # convert protobuff to ir_graph
    graph = pb_to_ir.convert(sys.argv[1])

    # export ir_graph's config
    config = exportConfig(graph)
    # config = exportConfig_json(graph)

    # save config file
    save_config(config, "test.cfg")

    # convrt config file to ir_graph
    test_graph = importConfig("test.cfg")

    model2 = ir_to_pb.convert(test_graph)
    onnx.save(model2, "test2")


