import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
import numpy as np

class transpose_into_reshape():

    def match_conditions(self, node):
        if node.op_type == "Transpose" and len(node.next_node) ==1 and node.next_node[0].op_type == "Reshape":
            for attr in node.attribute:
                # tranpose 参数[0,2,3,1]的情况  且 input_data 
                if len(node.input[0].dims) == 4 and attr.name == "perm" and attr.data == [0,2,3,1] : 
                    logger.debug("transpose %s, pem %s", node.input[0].dims, attr.data)
                    [a,b,c,d] = node.input[0].dims
                    # reshape的outoput是（1，x）的情况
                    size_1_num = 0
                    for i in [b,c,d]: # 在变化的b,c,d维度中，其中2个size都为1，那么不影响下个reshape的结果
                        if i == 1:
                           size_1_num = size_1_num +1
                    if size_1_num == 2:
                        return True
                elif len(node.input[0].dims) == 5 and attr.name == "perm" and attr.data == [0,1,2,4,3] :
                    logger.warn("transpose %s, pem %s", node.input[0].dims, attr.data)
                    [a,b,c,d,e] = node.input[0].dims
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
                logger.warn("---- transpose into reshape %s, %s", node.name, node.input[0].dims)
                # 删除无效的transpose
                for node2 in node.next_node:
                    node2.input = node.input
                ir_graph.node_list.remove(node)
                return True

        return False