import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
from IR import pb_to_ir

# 判断是否满足条件
def match_conditions(node):
    if node.op_type == "ReduceMean":
        # axes=[2,3] （除0,1以外的全部维度）
        input_len = len(node.input[0].dims)
        axes = []
        for attr in node.attribute:
            if attr.name == "axes":
                axes = attr.data
                temp = list(range(2, input_len))
                # print("input_shape:", node.input[0].dims)
                # print("input_shape_len:", input_len)
                # print("axes:", axes)
                # print("temp:", temp)

                if axes == temp:
                    return True
                else:
                    print("can not convert axes=", axes)
                    return False

    return False


# 只允许单列表插入
def add_one_node(node, type_str):

    new_output = ir.Value()
    new_output.name =  node.output[0].name + "_" + type_str
    new_output.data_type = node.output[0].data_type

    new_node = ir.Node()
    new_node.op_type = type_str
    new_node.pre_node = [node]
    new_node.next_node = node.next_node
    new_node.input = node.output[0]
    new_node.output.append(new_output)

    node.next_node[0].input = new_node.output
    node.next_node[0].pre_node =[new_node]
    
    return new_node


def run_pass(graph):  
    for node in graph.node_list:
        if match_conditions(node) == True:
            print("---- convert reduceMean to globalAveragePool.", node.output[0].name)
            
            # 获取input shape 以及 axis属性
            input_shape = node.input[0].dims
            axes = []
            keepdims = 1
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = attr.data
                if attr.name == "keepdims":
                    keepdims = attr.data[0]

            print("input_shape:", input_shape)
            print("axes:", axes)
            print("keepdims:", keepdims)

            # 替换为GlobalAveragePool
            node.name = ""
            node.op_type = "GlobalAveragePool"
            node.attribute = []
            node.weight = []

            output_shape = [i for i in node.input[0].dims]
            for i in axes:
                output_shape[i] = 1
            node.output[0].dims = output_shape
                
            # keemdims =0 保持维度，需要进行降维度，添加reshape layer.
            if keepdims == 0:
                # 创建reshape的参数 init
                new_weight = ir.Value()
                new_weight.name =  node.output[0].name + "_reshape_weight"
                new_weight.dims = [2]
                new_weight.data = [input_shape[0], input_shape[1]]
                new_weight.data_type = 7 # INT64

                # 创建reshape的output
                new_output = ir.Value()
                new_output.name =  node.output[0].name + "_reshape"
                new_output.dims = [input_shape[0], input_shape[1]]
                new_output.data_type = node.output[0].data_type

                # 创建reshape node
                new_node = ir.Node()
                new_node.op_type = "Reshape"
                new_node.weight.append(new_weight)
                new_node.input = node.output
                new_node.output.append(new_output)

                # 必须insert再node.next_node[0]之前
                index = graph.node_list.index(node.next_node[0])
                graph.node_list.insert(index, new_node)
                node.next_node[0].input = [new_output]
                print("add reshape node after globalAveragePool")

            # 更新graph
            pb_to_ir.updata_graph(graph)
            print("convert to globalAveragePool success.")

            return False

    return True

def run(graph):
    finish_flag = False
    while finish_flag == False :
        finish_flag = run_pass(graph)
