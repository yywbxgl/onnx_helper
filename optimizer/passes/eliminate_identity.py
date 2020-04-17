import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase 


class eliminate_identity(PassCase):
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Identity":
            # todo 支持并行的node删除
            if len(node.pre_node) <=1 and len(node.next_node)<=1:
                return True
            else:
                logger.warn("can not eliminate node. %s", node.op_type)
        return False

    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- eliminate node %s %s", node.op_type, node.output[0].name)
                # 如果不是 last_node,那么需要修改next_node.input
                if node.output[0].name != ir_graph.output.name:
                    for node2 in node.next_node:
                        node2.input = node.input
                else:
                # 如果是last_node, 那么需要修改pre_node.output
                    for node2 in node.pre_node:
                        node2.output = node.output

                # 删除当前node
                ir_graph.node_list.remove(node)
                return True
                
        return False


