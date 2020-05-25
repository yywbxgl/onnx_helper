import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
import numpy as np

class transpose_eliminate():

    def match_conditions(self, node):                
        return False

    def get_match_transpose(self, ir_graph):
        find_flag_1 = find_flag_2 = False
        for node in ir_graph.node_list:
            if node.op_type == "Transpose":
                for attr in node.attribute:
                    # todo 暂时只支持transpose 一个输出
                    if len(node.input[0].dims) == 4 and attr.name == "perm" and attr.data == [0,2,3,1] and len(node.next_node) == 1:
                        input_shape_1 = node.input[0].dims
                        output_shape_1 = node.output[0].dims
                        find_flag_1 = True
                        find_node_1 = node.name
                    elif len(node.input[0].dims) == 4 and attr.name == "perm" and attr.data == [0,3,1,2] and len(node.next_node) == 1:
                        input_shape_2 = node.input[0].dims
                        output_shape_2 = node.output[0].dims
                        find_flag_2 = True
                        find_node_2 = node.name

            if find_flag_1 == True and find_flag_2 == True:
                break

        if find_flag_1 == True and find_flag_2 == True and input_shape_1 == output_shape_2 and output_shape_1 == input_shape_2:
            logger.debug("find match tranpose %s %s", find_node_1, find_node_2)
            return [find_node_1, find_node_2]

        return []


    def run_pass(self, ir_graph):
        names = self.get_match_transpose(ir_graph)
        if len(names) != 0 :
            logger.info("---- transpose eliminate %s", names)
            for name in names:
                for node in ir_graph.node_list:
                    if node.name == name:
                        # 删除transpose
                        for node2 in node.next_node:
                            node2.input = node.input
                        ir_graph.node_list.remove(node)
            return True

        return False