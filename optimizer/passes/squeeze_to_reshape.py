import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
import logging
logger = logging.getLogger(__name__)

from IR import ir


class squeeze_to_reshape():
        
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Squeeze":
            # 需要知道上一层的形状
            for i in node.input:
                if len(i.dims) == 0:
                    logger.warn("input shape unkown. %s", i.dims)
                    return False

            for attr in node.attribute:
                if attr.name == "axes":
                    for i in attr.data:
                        if node.input[0].dims[i] != 1:  # 只有一个输入
                            logger.warn("can not remove shape %s, axes %s",  i.dims, i.data)
                            return False
            return True
        return False


    def run_pass(self, graph):  
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                # 获取input shape 以及 axis属性
                input_shape = node.input[0].dims
                axis = []
                for attr in node.attribute:
                    if attr.name == "axes":
                        axis = attr.data
                logger.info("---- convert squeeze to reshape. %s  %s  %s", node.output[0].name, input_shape, axis)

                # 计算reshape参数
                output_shape = []
                for i in range(len(input_shape)):
                    if i not in axis:
                        output_shape.append(input_shape[i])               
                
                # 替换Flatten node
                new_weight = ir.Value()
                new_weight.name =  node.pre_node[0].output[0].name + "_reshape"
                new_weight.dims = [len(output_shape)]
                new_weight.data = output_shape
                new_weight.data_type = 7 # INT64

                node.name = ""
                node.op_type = "Reshape"
                node.attribute = []
                node.weight = []
                node.weight.append(new_weight)

                return True

        return False

