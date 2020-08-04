import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer import common

# 删除无效的reshape  输入形状和输出相同
class reshape_nop_eliminate():
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Reshape":
            if len(node.input[0].dims) != 0 and len(node.output[0].dims) != 0 :
                if node.input[0].dims == node.output[0].dims:
                    print("------", node.input[0].dims , node.output[0].dims)
                    return True
            else:
                logger.warn("can not get reshape input shape and output shape.")

        return False


    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- eliminate node %s %s", node.op_type, node.output[0].name)
                ir_graph = common.eliminate_node(ir_graph, node.name)
                return True

        return False


