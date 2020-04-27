import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase
import numpy as np
import copy

class softmax_swap_down(PassCase):

    def match_conditions(self, node):
        if node.op_type == "Softmax":
            if len(node.pre_node) == 1 and len(node.next_node) == 1:
                return True 
                
        return False

    def sawp_node(self, node1, node2):

        node_temp = copy.copy(node2) # 浅拷贝，可变对象依旧是引用
  
        node2.name = node1.name
        node2.op_type = node1.op_type
        node2.attribute = node1.attribute
        node2.weight = node1.weight

        node1.name = node_temp.name
        node1.op_type = node_temp.op_type
        node1.attribute = node_temp.attribute
        node1.weight = node_temp.weight
        node1.output[0].dims = node2.output[0].dims


    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node) == True:
                logger.warn("---- Softmax move %s,  %s", node.name, node.next_node[0].name)
                self.sawp_node(node, node.next_node[0])
                return True

        return False