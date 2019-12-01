import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import collections

from IR import ir

def exportConfig(ir_graph):
    print("----------------------")
    config = "input: " + ir_graph.input.name + "  " + str(ir_graph.input.dims) + "\n"
    config += "output: " + ir_graph.output.name + "  " + str(ir_graph.output.dims) + "\n"

    for i in ir_graph.node_list:
        config += "{\n"
        config += "\tname: " + i.name + "\n"

        for n in range(len(i.input)):
            if n == 0:
                config += "\tinputs: " + i.input[n].name
            else:
                config += "  " + i.input[n].name
        # 每个node不一定有input
        if (len(i.input) != 0):
            config += "\n"

        for n in range(len(i.output)):
            if n == 0:
                config += "\toutputs: " + i.output[n].name
            else:
                config += "  " + i.output[n].name
        # 每个node必须指定output
        config += "\n"

        for w in i.weight:
            config += "\tweight: " + w.name + "  " + str(w.dims) + "\n"

        for attr in i.attribute:
            config += "\tattr: " + attr.name + "  " + str(attr.data) + "\n"

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

        # if len(node.attribute) != 0:
        #     temp["attribute"] = []
        #     for attr in node.attribute:
        #         # todo  attr vlaue 值为tensor或者graph时
        #         if attr.data_type in [4, 5, 9, 10]:
        #             print("can not parse attr data", attr.name, attr.data_type)
        #         temp["attribute"].append({"name":attr.name, "value":attr.data})
        
        if len(node.weight) != 0:
            temp["weights"] = []
            for w in node.weight:
                print(type(w.dims))
                temp["weights"].append(collections.OrderedDict({"name":w.name, "dims":w.dims}))

        config["node"].append(temp)

    config_json = json.dumps(config, indent=4)
    print(config_json)
    return config



test_config = {
    "input" : {
        "name": "INPUT",
        "dims": [1,3,224,224]
    },
    "output": {
        "name": "OUTPUT",
        "dims": [1,1000]
    },
    "node": [
        {
            "name": "conv_1",
            "op_type": "Conv",
            "inputs": ["INPUT"],
            "outputs": ["conv_1_out"],
            "weights": [ 
                {"name": "conv_1_w", "dims":[3,3]},
                {"name": "conv_1_b", "dims":[3]},
            ],
            "attribute": [
                {"name": "kernal_shape", "value":[3,3]} ,
                {"name": "stride", "value":[2,2]}
            ]
        },
        {
            "name": "conv_2",
            "op_type": "Conv",
            "inputs": ["conv_1_out"],
            "outputs": ["conv_2_out"],
            "weights": [ 
                {"name": "conv_2_w", "dims":[3,3]},
                {"name": "conv_2_b", "dims":[3]},
            ],
            "attribute": [
                {"name": "kernal_shape", "value":[3,3]} ,
                {"name": "stride", "value":[2,2]}
            ]
        }
    ]
}

if __name__ == "__main__":

    from IR import pb_to_ir

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " OnnxModel")
        sys.exit(-1)


    # print(test_config)
    # js = json.dumps(test_config, indent=2)
    # print(js)
    # sys.exit(-1)

    graph = pb_to_ir.convert(sys.argv[1])
    exportConfig_json(graph)