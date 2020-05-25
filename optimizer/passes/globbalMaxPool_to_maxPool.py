import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
import logging
logger = logging.getLogger(__name__)


class globbalMaxPool_to_maxPool():

    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "GlobalMaxPool":
            for i in node.input:
                if len(i.dims) == 0:
                    logger.warn("input shape unkown. %s", i.dims)
                    return False
            return True

        return False


    def run_pass(self, graph):  
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- globbalMaxPool_to_maxPool. %s", node.output[0].name)
                
                # 获取input shape 
                input_shape = node.input[0].dims
                kernal_size = [input_shape[2], input_shape[3]]

                # 修改node属性
                node.name = ""
                node.op_type = "MaxPool"
                node.attribute = []

                attr_temp = ir.Value()
                attr_temp.name = "pads"
                attr_temp.dims = [4]
                attr_temp.data = [0,0,0,0]
                attr_temp.data_type = 7
                node.attribute.append(attr_temp)

                attr_temp = ir.Value()
                attr_temp.name = "kernel_shape"
                attr_temp.dims = [2]
                attr_temp.data = [7,7]
                attr_temp.data_type = 7
                node.attribute.append(attr_temp)

                attr_temp = ir.Value()
                attr_temp.name = "strides"
                attr_temp.dims = [2]
                attr_temp.data = [1,1]
                attr_temp.data_type = 7
                node.attribute.append(attr_temp)

                logger.warn("input shape. %s  %s", input_shape, kernal_size)

                return True

        return False