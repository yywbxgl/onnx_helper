import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir

# 删除无效的slice 输入形状和输出相同
class slice_nop_eliminate():
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Slice":

            # 如果input&outpu shape相同 直接返回True
            if len(node.input[0].dims) !=0 and len(node.output[0].dims)!=0:
                if node.input[0].dims == node.output[0].dims:
                    return True

            # 通过参数判断
            # for i in node.input:
            #     if len(i.dims) == 0:
            #         logger.warn("input shape unkown. %s", i.dims)
            #         return False

            # for i in node.weight:
            #     if len(i.dims) == 0:
            #         logger.warn("input shape unkown. %s", i.dims)
            #         return False

            # inputs = []
            # inputs.extend(node.input)
            # inputs.extend(node.weight)
            
            # data_shape = inputs[0].dims
            # starts = inputs[1].data
            # ends = inputs[2].data
            # axes = inputs[3].data
            # steps = inputs[4].data

            # for i in steps:
            #     if i != 1:
            #         return False
            
            # for i in starts:
            #     if i != 0:
            #         return False
            
            # for index in range(len(axes)):
            #     if ends[index] < data_shape[axes[index]]:
            #         return False

            # return True
                
        return False


    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- eliminate slice node %s %s", node.op_type, node.output[0].name)
                # 如果不是 last_node,那么需要修改next_node.input
                if node.output[0].name not in [t.name for t in ir_graph.output]:
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


