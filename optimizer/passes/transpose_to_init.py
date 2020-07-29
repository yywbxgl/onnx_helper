import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from IR import convert_utils
import numpy as np

class transpose_to_init():

    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Transpose":
            # 需要知道上一层的是否为静态张量
            for i in node.input:
                if i.init == False:
                    return False
            return True

        return False


    def run_pass(self, graph):
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- convert Transpose to initiliazer. %s", node.output[0].name)

                input_data = []
                if len(node.input) < 1:
                    input_data = convert_utils.convert_raw_data(node.weight[0])
                else:
                    input_data = convert_utils.convert_raw_data(node.input[0])

                attr_pem = []
                for attr in node.attribute:
                    if attr.name == "perm" :
                        attr_pem = attr.data

                logger.info("input_data size: %s  attr_pem: %s", input_data.dims, attr_pem)

                data = np.array(input_data.data)
                data = np.reshape(data, input_data.dims)
                y = np.transpose(data, attr_pem)       

                
                # 保存结果到 到initilizer
                for i in node.next_node[0].input:
                    if i.name == node.output[0].name:
                        i.dims = list(y.shape)
                        i.data = y.tolist()
                        if type(i.data) != type([]):
                            i.data = [i.data]
                        i.data_type = node.output[0].data_type
                        i.init = True

                node.next_node[0].pre_node = []

                # 删除gateher node
                graph.node_list.remove(node)
                return True
    
        return False