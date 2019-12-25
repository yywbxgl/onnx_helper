import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import numpy as np

from IR import ir
from IR import convert_utils

# 判断是否满足条件
def match_conditions(node):
    if node.op_type == "Unsqueeze":
        # 需要知道上一层的是否为静态张量
        for i in node.input:
            if i.init == False:
                return False
        return True

    return False


def run_pass(graph):
    for node in graph.node_list:
        if match_conditions(node) == True:
            print("---- convert unsqueeze to initiliazer.", node.output[0].name)
            input_data = convert_utils.get_raw_data(node.input[0])
            print("input_data:", node.input[0].data)
            
            axis_arg = 0
            for i in node.attribute:
                if i.name == "axes":
                    axis_arg = i.data
            print("axes=", axis_arg)

            y = np.array(input_data)
            for i in axis_arg:
                y = np.expand_dims(y, axis=i)

            print("unsqueze output :", y)

            # 保存结果到 到initilizer
            for i in node.next_node[0].input:
                if i.name == node.output[0].name:
                    i.dims = list(y.shape)
                    if y is not list:
                        y = [y]
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





