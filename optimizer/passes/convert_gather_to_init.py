import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import numpy as np

from IR import ir

# 判断是否满足条件
def match_conditions(node):
    if node.op_type == "Gather":
        # 需要知道上一层的是否为静态张量
        for i in node.input:
            if i.init == False:
                return False
        return True

    return False


def run_pass(graph):
    for node in graph.node_list:
        if match_conditions(node) == True:
            print("---- convert gather to initiliazer.", node.output[0].name)
            print("input_data:", node.input[0].data)
            print("input_indeices:", node.input[1].data)

            axis_arg = 0
            for i in node.attribute:
                if i.name == "axis":
                    axis_arg = int(i.data[0])

            data = np.array(node.input[0].data)
            indices = np.array(node.input[1].data)
            y = np.take(data, indices, axis=axis_arg)
            print("gather output shape:", list(y.shape))

            # 保存结果到 到initilizer
            for i in node.next_node[0].input:
                if i.name == node.output[0].name:
                    i.dims = list(y.shape) 
                    if len(i.dims) == 1 and i.dims[0] == 1:  # 转为标量？？
                        i.dims = []
                    i.data = list(y)
                    i.data_type = node.input[0].data_type
                    i.init = True

            node.next_node[0].pre_node = []

            # 删除gateher node
            graph.node_list.remove(node)
            return False
    
    return True

      
def run(graph):
    finish_flag = False
    while finish_flag == False :
        finish_flag = run_pass(graph)    





