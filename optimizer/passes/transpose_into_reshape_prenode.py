import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase
import numpy as np

class transpose_into_reshape_prenode(PassCase):

    def match_conditions(self, node):
        if node.op_type == "Reshape" and len(node.next_node) ==1 and node.next_node[0].op_type == "Transpose":
            for attr in node.next_node[0].attribute:
                # tranpose 参数[0,2,3,1]的情况  且 input_data 
                if len(node.next_node[0].input[0].dims) == 4 and attr.name == "perm" and attr.data == [0,3,1,2] : 
                    logger.debug("transpose %s, pem %s", node.next_node[0].input[0].dims, attr.data)
                    [a,b,c,d] = node.next_node[0].input[0].dims
                    # reshape的outoput是（1，x）的情况
                    size_1_num = 0
                    for i in [b,c,d]: # 在变化的b,c,d维度中，其中2个size都为1，那么不影响下个reshape的结果
                        if i == 1:
                           size_1_num = size_1_num +1
                    if size_1_num == 2:
                        return True
                elif len(node.next_node[0].input[0].dims) == 5 and attr.name == "perm" and attr.data == [0,1,2,4,3] :
                    logger.warn("transpose %s, pem %s", node.next_node[0].input[0].dims, attr.data)
                    [a,b,c,d,e] = node.next_node[0].input[0].dims
                    # reshape的outoput是（1，x）的情况
                    size_1_num = 0
                    for i in [d,e]: # 在变化的d,e维度中，其中1个size为1，那么不影响下个reshape的结果
                        if i == 1:
                            size_1_num = size_1_num +1
                    if size_1_num == 1:
                        return True
        return False


    def run_pass(self, ir_graph):
        # logger.warn("check transpose_into_reshape")
        for node in ir_graph.node_list:
            if self.match_conditions(node) == True:
                logger.warn("---- transpose into pre reshape  %s, %s", node.name, node.input[0].dims)
                # 修改reshape参数
                logger.warn(node.weight[0].name)
                logger.warn(node.weight[0].dims)
                logger.warn(node.weight[0].data)
                logger.warn(node.weight[0].data_type)
                logger.warn(node.weight[0].raw)
                logger.warn(node.weight[0].init)

                node.weight[0].dims = [len(node.next_node[0].output[0].dims)]
                node.weight[0].data = node.next_node[0].output[0].dims
                node.weight[0].raw = False
                node.weight[0].data_type = 7

                logger.warn(node.weight[0].name)
                logger.warn(node.weight[0].dims)
                logger.warn(node.weight[0].data)
                logger.warn(node.weight[0].data_type)
                logger.warn(node.weight[0].raw)
                logger.warn(node.weight[0].init)

                # 删除无效的transpose
                node.next_node[0].next_node[0].input[0] = node.output[0]
                ir_graph.node_list.remove(node.next_node[0])
                return True

        return False