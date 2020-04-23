import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase
import numpy as np

class transpose_into_reducemean(PassCase):

    def match_conditions(self, node):
        if node.op_type == "Transpose" and len(node.next_node) == 1 and node.next_node[0].op_type == "ReduceMean":
            for attr in node.attribute:
                # tranpose 参数[0,2,3,1]的情况  且 input_data 
                if attr.name == "perm" and attr.data == [0,2,3,1] :
                    logger.info("transpose %s, pem %s", node.input[0].dims, attr.data)
                    for attr2 in node.next_node[0].attribute:
                        # 限制到参数[1,2]
                        if attr2.name == "axes"  and attr2.data == [1,2]: 
                            return True
                
        return False


    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- transpose into reduceMean %s, %s, ", node.name, node.input[0].dims)
                # 修改reduceMean的参数
                for attr2 in node.next_node[0].attribute:
                    if attr2.name == "axes":
                        attr2.data = [2,3]
                
                # 删除transpose
                for node2 in node.next_node:
                    node2.input = node.input
                ir_graph.node_list.remove(node)
                return True

        return False