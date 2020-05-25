import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
from IR import pb_to_ir
from IR import convert_utils
import copy
import logging
logger = logging.getLogger(__name__)


class convert_constant_to_init():
        
    def match_conditions(self, node):
        if node.op_type == "Constant":
            return True
        return False


    def run_pass(self, graph):
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- convert constant to init. %s",  node.output[0].name)
                if (len(node.attribute) == 1 and node.attribute[0].name == "value"):
                    temp = pb_to_ir.protoTensor_to_irValue(node.attribute[0].data[0])
                    # 保存的int不使用raw_data
                    temp = convert_utils.convert_raw_data(temp)

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






