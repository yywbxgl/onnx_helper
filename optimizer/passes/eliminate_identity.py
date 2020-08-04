import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer import common
import copy


class eliminate_identity():
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Identity":
            # todo 支持并行的node删除
            if len(node.pre_node) <=1 :
                return True
            else:
                logger.warn("can not eliminate node. %s", node.op_type)
        return False

    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- eliminate node %s %s", node.op_type, node.output[0].name)
                ir_graph = common.eliminate_node(ir_graph, node.name)
                return True
                
        return False


