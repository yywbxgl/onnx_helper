import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
import numpy as np

class transpose_input():
    def match_conditions(self, node, input_names):
        # input 接一个tranpose的情况  修改input
        if node.op_type == "Transpose" and node.input[0].name in input_names:
            # logger.warn("---- node.name:%s, input_name:%s", node.input[0].name, input_name)
            for attr in node.attribute:
                # tranpose 参数[0,3,1,2]的情况
                if attr.name == "perm" and attr.data == [0,3,1,2]:
                    return True
        return False


    def run_pass(self, ir_graph):
        input_names = [i.name for i in ir_graph.input]
        for node in ir_graph.node_list:
            if self.match_conditions(node, input_names):

                for i in ir_graph.input:
                    if node.input[0].name == i.name:

                        logger.info("--- transpose_input: %s", i.dims)
                        # 1. 修改input的shape
                        [a,b,c,d] = i.dims
                        logger.debug("---- input_shape:%s", [a,b,c,d])
                        # transposed = np.transpose(input_shape, [0,3,1,2])
                        i.dims = [a,d,b,c]
                        logger.debug("ir_graph.input.dims, %s", i.dims)

                        # 2. 删除transpose node
                        for node2 in node.next_node:
                            node2.input = node.input
                        ir_graph.node_list.remove(node)
                        return True

        return False