import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import numpy as np
import logging
logger = logging.getLogger(__name__)

from IR import ir
from IR import convert_utils


class convert_unsuqeeze_to_init():

    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Unsqueeze":
            # 需要知道上一层的是否为静态张量
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
                logger.info("---- convert unsqueeze to initiliazer. %s", node.output[0].name)
                if len(node.input) != 0 :
                    input_data = convert_utils.get_raw_data(node.input[0])
                else:
                    input_data = convert_utils.get_raw_data(node.weight[0])
                logger.info("input_data: %s", input_data)
                
                axis_arg = 0
                for i in node.attribute:
                    if i.name == "axes":
                        axis_arg = i.data
                logger.info("axes=%s", axis_arg)

                y = np.array(input_data)
                for i in axis_arg:
                    y = np.expand_dims(y, axis=i)

                logger.info("unsqueze output : %s  shape:%s", y, y.shape)

                # 保存结果到 到initilizer
                for i in node.next_node[0].input:
                    if i.name == node.output[0].name:
                        i.dims = list(y.shape)
                        i.data = y.tolist()
                        if type(i.data) != type([]):
                            i.data = [i.data]
                        if len( node.input) != 0:
                            i.data_type = node.input[0].data_type
                        else:
                            i.data_type = node.weight[0].data_type
                        i.init = True

                node.next_node[0].pre_node = []

                # 删除gateher node
                graph.node_list.remove(node)
                return True
        
        return False




