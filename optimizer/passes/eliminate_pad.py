import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer import common


class eliminate_pad():
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Pad":
            pads = []
            for a in node.attribute:
                if a.name == "pads":
                    pads = a.data
            
            for i in pads:
                if (i != 0):
                    return False

            # print(node.output[0].name, "has invalid pads", pads)
            return True
        return False


    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- eliminate node %s %s", node.op_type, node.output[0].name)
                ir_graph = common.eliminate_node(ir_graph, node.name)
                return True

        return False

