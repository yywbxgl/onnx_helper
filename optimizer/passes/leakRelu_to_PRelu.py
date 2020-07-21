import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
import logging
logger = logging.getLogger(__name__)

from IR import ir

class leakRelu_to_PRelu():
     # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "LeakyRelu":
            return True
        return False

    def run_pass(self, graph):  
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                alpha = 0.01 # defalut value
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = attr.data[0]

                # 创建一个init 存放slope tensor
                new_weight = ir.Value()
                new_weight.name =  node.pre_node[0].output[0].name + "_slope"
                new_weight.dims = [1]
                new_weight.data = [alpha]
                new_weight.data_type = 1 # FLOAT64

                node.name = ""
                node.op_type = "PRelu"
                node.attribute = []
                node.weight = []
                node.weight.append(new_weight)

                logger.info("leakReulu  convert to PRelu. alpha=%s", alpha)

                return True

        return False
