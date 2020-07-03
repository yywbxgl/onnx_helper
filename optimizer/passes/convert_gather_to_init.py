import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import numpy as np
import logging
logger = logging.getLogger(__name__)

from IR import ir
from IR import convert_utils


class convert_gather_to_init():

    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Gather":
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
                logger.info("---- convert gather to initiliazer. %s", node.output[0].name)
                
                # if node.input[0].raw == True:
                #     input_data_0 = convert_utils.convert_raw_data(node.input[0]).data
                # else:
                #     input_data_0 = node.input[0].data

                # if node.input[1].raw == True:
                #     input_data_1 = convert_utils.convert_raw_data(node.input[1]).data
                # else:
                #     input_data_1 = node.input[1].data

                # if node.input[1].dims == []:
                #     input_data_1 = input_data_1[0]

                input_data_0 = convert_utils.get_raw_data(node.input[0])

                if len(node.input) <2:
                    input_data_1 =  convert_utils.get_raw_data(node.weight[0])
                else:
                    input_data_1 = convert_utils.get_raw_data(node.input[1])
                logger.info("input_data: %s", input_data_0)
                logger.info("input_indeices: %s", input_data_1)

                axis_arg = 0
                for i in node.attribute:
                    if i.name == "axis":
                        axis_arg = int(i.data[0])

                data = np.array(input_data_0)
                indices = np.array(input_data_1).astype(np.int64)
                y = np.take(data, indices, axis=axis_arg)
                logger.info("gather output: %s", y)

                # 保存结果到 到initilizer
                for i in node.next_node[0].input:
                    if i.name == node.output[0].name:
                        i.dims = list(y.shape)
                        i.data = y.tolist()
                        if type(i.data) != type([]):
                            i.data = [i.data]
                        i.data_type = node.input[0].data_type
                        i.init = True

                node.next_node[0].pre_node = []

                # 删除gateher node
                graph.node_list.remove(node)
                return True
        
        return False




