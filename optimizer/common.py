import sys, os

import logging
logger = logging.getLogger(__name__)

# node删除， 支持 1个input  多个output 
# 或者 多个input  1个output

def eliminate_node(graph, node_name):
    for node in graph.node_list:
        if node.name == node_name:

            # 同时有多个input 和ouput 无法直接删除
            if len(node.input) > 1 or len(node.output) > 1 :
                logger.error("can not elimnate node. %s has multi input or output", node_name)
                sys.exit()

            # 如果不是 last_node,那么需要修改next_node.input
            if node.output[0].name not in [t.name for t in graph.output]:
                for node2 in node.next_node:
                    for i in node2.input:
                        if i.name == node.output[0].name:
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
            graph.node_list.remove(node)
            return graph

    return graph


