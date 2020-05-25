import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir


class reshape_consecutive_eliminate():
    # 判断是否满足条件
    def match_conditions(self, node):
        # logger.warn("------------------%s  %s  %s", node.name, node.op_type, len(node.next_node) )
        if node.op_type == "Reshape" and len(node.next_node) == 1:
            # 删除连续的reshape
            if node.next_node[0].op_type == "Reshape":
                return True
        return False


    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.warn("---- eliminate node %s %s", node.op_type, node.output[0].name)
                # 如果不是 last_node,那么需要修改next_node.input
                if node.output[0].name != ir_graph.output.name:
                    for node2 in node.next_node:
                        for i in node2.input:
                            if i.name == node.output[0].name:
                                # i = copy.deepcopy(node.input[0])
                                i.name  =  node.input[0].name
                                i.dims = node.input[0].dims
                                i.data = node.input[0].data
                                i.data_type = node.input[0].data_type
                                i.raw = node.input[0].raw
                                i.init = node.input[0].init
                else:
                # 如果是last_node, 那么需要修改pre_node.output
                    for node2 in node.pre_node:
                        node2.output = node.output

                # 删除当前node
                ir_graph.node_list.remove(node)
                return True

        return False


