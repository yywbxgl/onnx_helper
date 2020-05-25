import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
import logging
logger = logging.getLogger(__name__)

from IR import ir


class convert_flatten_to_reshape():
        
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Flatten":
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
                # 获取input shape 以及 axis属性
                input_shape = node.input[0].dims
                axis = 1
                if (len(node.attribute) != 0):
                    axis = node.attribute[0].data[0]
                logger.info("---- convert flatten to reshape. %s", node.output[0].name)

                if (axis < 0):
                    logger.warn("not support nagative axis")
                    sys.exit(-1)

                # 计算reshape参数
                out_1 = 1
                out_2 = 1 
                if axis == 0:
                    # When axis = 0, the shape of the output tensor is (1, (d_0 X d_1 ... d_n)
                    for i in input_shape:
                        out_2 *= i
                else:
                    for i in range(axis):
                        out_1 *= input_shape[i]
                    for i in range(axis, len(input_shape)):
                        out_2 *= input_shape[i]
                
                # 替换Flatten node
                new_weight = ir.Value()
                new_weight.name =  node.pre_node[0].output[0].name + "_reshape"
                new_weight.dims = [2]
                new_weight.data = [out_1, out_2]
                new_weight.data_type = 7 # INT64

                node.name = ""
                node.op_type = "Reshape"
                node.attribute = []
                node.weight = []
                node.weight.append(new_weight)

                logger.info("input=%s axis=%d  convert to reshape [%d %d]", str(input_shape), axis, out_1, out_2)

                return True

        return False

