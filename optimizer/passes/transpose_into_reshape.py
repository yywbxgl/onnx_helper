import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase
import numpy as np

class transpose_into_reshape(PassCase):

    def match_conditions(self, node):
        if node.op_type == "Transpose":
            for attr in node.attribute:
                # tranpose 参数[0,2,3,1]的情况  且 input_data 
                if attr.name == "perm" and attr.data == [0,2,3,1] and node.next_node[0].op_type == "Reshape":
                    logger.debug("transpose %s, pem %s", node.input[0].dims, attr.data)
                    # reshape的shape参数为(0,-1) 表示压缩成1维数据
                    [a,b,c,d] = node.input[0].dims
                    if [a,c,d] == [1,1,1] and node.next_node[0].weight[0].data == [0,-1]:
                        return True
                
        return False


    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- transpose into reshape %s, %s, ", node.name, node.input[0].dims)
                # 删除无效的transpose
                for node2 in node.next_node:
                    node2.input = node.input
                ir_graph.node_list.remove(node)
                return True

        return False