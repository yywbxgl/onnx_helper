import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
import logging
logger = logging.getLogger(__name__)


class convert_shape_to_init():

    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Shape":
            # 需要知道上一层的形状
            for i in node.input:
                if len(i.dims) == 0:
                    logger.warn("input shape unkown. %s", i.dims)
                    return False
            return True
        return False


    def run_pass(self, graph):
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                input_shape = node.input[0].dims
                logger.info("---- convert shape to initiliazer. %s", node.output[0].name)
                logger.info("input_shape: %s", input_shape)

                # 保存shape 到initilizer
                for i in node.next_node[0].input:
                    if i.name == node.output[0].name:
                        i.dims = [len(input_shape)]
                        i.data = input_shape
                        i.data_type = 7
                        i.init = True

                node.next_node[0].pre_node = []

                # 删除node
                graph.node_list.remove(node)
                return True
        
        return False


