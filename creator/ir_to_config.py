import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import collections
import numpy as np

from IR import ir

def exportConfig(ir_graph):
    print("------- dump ir config ---------")

    split_str = " ; "

    config = "graph : " + ir_graph.name + "\n"
    for i in ir_graph.input:
        if i.data_type == ir.DataType.FLOAT.value:
            config += "input : " + i.name + split_str + str(i.dims) + "\n"
        else:
            config += "input : " + i.name + split_str + str(i.dims) + split_str +  ir.DataType(i.data_type).name +"\n"

    for i in ir_graph.output:
        if i.data_type == ir.DataType.FLOAT.value:
            config += "output : " + i.name + split_str + str(i.dims) + "\n"
        else:
            config += "output : " + i.name + split_str + str(i.dims) + split_str +  ir.DataType(i.data_type).name + "\n"

    config += "ir_version : " + str(ir_graph.ir_version) +  "\n"
    config += "opset : " + str(ir_graph.opset) +  "\n"

    for i in ir_graph.node_list:
        config += "{\n"
        config += "\tname: " + i.name + "\n"
        config += "\ttype: " + i.op_type + "\n"

        for n in range(len(i.input)):
            if n == 0:
                config += "\tinputs: " + i.input[n].name
            else:
                config += split_str + i.input[n].name
        # 每个node不一定有input
        if (len(i.input) != 0):
            config += "\n"

        for n in range(len(i.output)):
            if n == 0:
                config += "\toutputs: " + i.output[n].name
            else:
                config += split_str + i.output[n].name
        # 每个node必须指定output
        config += "\n"

        for w in i.weight:
            config += "\tweight: " + w.name + split_str + str(w.dims) 
            if w.raw == True and  w.data_type !=1 : # 当data为raw时候，保存data的数据类型，FLOAT为默认类型，不用保存
                w_type = ir.DataType(w.data_type)
                config += split_str +  w_type.name
            config += "\n"

        for attr in i.attribute:
            # todo parse tensors and graphs
            if attr.data_type in [5, 9, 10]:
                print("!!! can not parse attr. data type=", attr.data_type)
                sys.exit(-1)
            if len(attr.dims) != 0:
                config += "\tattr: " + attr.name + split_str + str(attr.data) + "\n"
            else:
                config += "\tattr: " + attr.name + split_str + str(attr.data[0]) + "\n"
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
                elif attr.data_type == 3: # 当数据类型为String时,转为字符串，否则无法转换为json
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


if __name__ == "__main__":

    from IR import pb_to_ir
    from IR import ir_to_pb

    # 导出onnx model 的config file 
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


