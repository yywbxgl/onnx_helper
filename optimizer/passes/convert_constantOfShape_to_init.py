import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
from IR import pb_to_ir
from IR import convert_utils
import copy
import logging
import numpy as np
logger = logging.getLogger(__name__)

# constantOfShape  value 为element值   input为output_tensor的形状

class convert_constantOfShape_to_init():
        
    def match_conditions(self, node):
        if node.op_type == "ConstantOfShape":
            for i in node.input:
                if i.init == False:
                    return False 
            for i in node.weight:
                if i.init == False:
                    return False 
            return True
        return False


    def run_pass(self, graph):
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                #todo  noinput or no attribute
                logger.info("---- convert ConstantOfShape to init. %s",  node.output[0].name)
                temp = ir.Value()
                temp.data = [0]
                temp.dims = [1]
                temp.data_type = 1 #FLOAT

                if (len(node.attribute) == 1 and node.attribute[0].name == "value"):
                    temp = pb_to_ir.protoTensor_to_irValue(node.attribute[0].data[0])
                    # 获取element
                    temp = convert_utils.convert_raw_data(temp)
                    element = temp.data[0]

                    # 扩展形状
                    if len(node.input) > 0 :
                        temp.dims = node.input[0].dims

                        temp.data = []
                        for i in temp.dims:
                            for j in range(i):
                               temp.data.append(element) 
                    elif len(node.weight) > 0 :
                        temp.dims = node.weight[0].dims
                        temp.data = []
                        for i in temp.dims:
                            for j in range(i):
                               temp.data.append(element) 
                        
                    logger.info("element:%s  shape:%s", element , temp.dims)

                    # 保存 constant 到initilizer
                    for i in node.next_node[0].input:
                        if i.name == node.output[0].name:
                            i.dims = temp.dims
                            i.data = temp.data
                            i.data_type = temp.data_type
                            i.raw = temp.raw
                            i.init = True

                    # 删除constant node
                    graph.node_list.remove(node)
                    return True

                else:
                    logger.error("error. can not support sparse_value")
                    sys.exit(-1)
                    return False
        return False






