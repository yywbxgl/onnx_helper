import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import numpy as np

from IR import ir
from IR import convert_utils

# 判断是否满足条件
def match_conditions(node):
    if node.op_type == "Concat":
        # 需要知道上一层的是否为静态张量
        for i in node.input:
            if i.init == False:
                return False
        return True

    return False


def run_pass(graph):
    for node in graph.node_list:
        if match_conditions(node) == True:
            print("---- convert concat to initiliazer.", node.output[0].name)

            input_all = []
            for i in node.input:
                input_data = convert_utils.get_raw_data(i)
                print("input_data:", input_data)
                input_all.append(np.asarray(input_data))
            
            # print("input_all:", input_all)

            axis_arg = 0
            for i in node.attribute:
                if i.name == "axis":
                    axis_arg = i.data
            print("axis=", axis_arg)

            y = np.concatenate(input_all, axis_arg[0])
            print("concat result :", y)

            # 保存结果到 到initilizer
            for i in node.next_node[0].input:
                if i.name == node.output[0].name:
                    i.dims = list(y.shape)
                    i.data = y.tolist()
                    if type(i.data) != type([]):
                        i.data = [i.data]
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





