import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase
import numpy as np
import copy

class transpose_swap(PassCase):

    def match_conditions(self, node):
        if node.op_type == "Transpose":
            for attr in node.attribute:
                # tranpose 参数[0,2,3,1]的情况   且只有一个output
                if len(node.input[0].dims) == 4 and attr.name == "perm" and attr.data == [0,2,3,1] and len(node.next_node) == 1:
                    # 下一个node不改变形状， 且只有一个output,
                    if len(node.next_node[0].next_node) ==1 and node.next_node[0].output[0].dims == node.output[0].dims:
                        # logger.debug("transpose move get %s, %s", node.name,  node.next_node[0].name)
                        return True 
                
        return False

    def sawp_node(self, node1, node2):

        node_temp = copy.copy(node2)
  
        node2.name = node1.name
        node2.op_type = node1.op_type
        node2.attribute = node1.attribute
        node2.weight = node1.weight
        node2.input[0].dims = node1.input[0].dims

        node1.name = node_temp.name
        node1.op_type = node_temp.op_type
        node1.attribute = node_temp.attribute
        node1.weight = node_temp.weight
        node1.output[0].dims = node_temp.input[0].dims

        logger.warn(node2.name)
        logger.warn(node1.name)


    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node) == True:
                logger.warn("---- transpose move %s,  %s", node.name, node.next_node[0].name)
                self.sawp_node(node, node.next_node[0])
                return True

        return False