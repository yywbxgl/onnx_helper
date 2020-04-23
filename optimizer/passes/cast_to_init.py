import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from IR import ir
from IR import pb_to_ir
from IR import convert_utils
import copy
import logging
logger = logging.getLogger(__name__)

from optimizer.optimizer import PassCase 


class cast_to_init(PassCase):
        
    def match_conditions(self, node):
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.data == [7]:
                        return True
                    else:
                        logger.warn("cast to %d not support.", attr.data)
        return False


    def run_pass(self, graph):
        for node in graph.node_list:
            if self.match_conditions(node) == True:
                logger.info("---- convert cast to init. %s",  node.output[0].name)

                # logger.debug(node.weight[0].name)
                # logger.debug(node.weight[0].raw)
                # logger.debug(node.weight[0].init)
                # logger.debug(node.weight[0].data_type)
                # logger.debug(node.weight[0].dims)
                # logger.debug(node.weight[0].data)

                # 保存的int不使用raw_data
                temp  = ir.Value()
                temp.name = node.weight[0].name
                temp.dims = node.weight[0].dims
                temp.raw = False
                temp.init = True
                temp.data = convert_utils.get_raw_data(node.weight[0])
                temp.data_type = 7 #INT64
                logger.debug(temp.data)

                # 保存cast 到initilizer
                for i in node.next_node[0].input:
                    if i.name == node.output[0].name:
                        i.name = temp.name
                        i.dims = temp.dims 
                        i.raw = temp.raw
                        i.init = temp.init
                        i.data = temp.data
                        i.data_type = temp.data_type

                # 删除constant node
                graph.node_list.remove(node)
                return True

        return False






