import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir

# 判断是否满足条件
def match_conditions(node):
    if node.op_type == "Shape":
        # 需要知道上一层的形状
        for i in node.input:
            if len(i.dims) == 0:
                print("warn. input shape unkown.", i.dims)
                return False
        return True
    return False


def run(graph):
    for node in graph.node_list:
        if match_conditions(node) == True:
            input_shape = node.input[0].dims
            print("---- convert shape to initiliazer.", node.output[0].name)
            print("input_shape:", input_shape)

            # 保存shape 到initilizer
            for i in node.next_node[0].input:
                if i.name == node.output[0].name:
                    i.dims = [len(input_shape)]
                    i.data = input_shape
                    i.data_type = 7
                    i.init = True

            node.next_node[0].pre_node = []

            # 删除node
            graph.node_list.remove(node)
            return False
    
    return True
            
            





