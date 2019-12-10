import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from IR import ir

# 判断是否满足条件
def match_conditions(node, operator_name):
    if node.op_type == operator_name:
        # todo 支持并行的node删除
        if len(node.pre_node) ==1 and len(node.next_node)== 1:
            return True
        else:
            print("can not eliminate node.", operator_name)
    return False


# 运行一次优化
def run(ir_graph, operator_name):

    for node in ir_graph.node_list:
        if match_conditions(node, operator_name):
            # 如果不是 last_node,那么需要修改next_node.input
            if node.output[0].name != ir_graph.output.name:
                for node2 in node.next_node:
                    node2.input = node.input
            else:
            # 如果是last_node, 那么需要修改pre_node.output
                for node2 in node.pre_node:
                    node2.output = node.output

            # 删除当前node
            print("eliminate_node", operator_name, node.name)
            ir_graph.node_list.remove(node)
            return False
            
    return True



