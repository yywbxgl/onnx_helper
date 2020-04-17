import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from IR import ir
from optimizer.optimizer import PassCase 


class fuse_pad_into_conv(PassCase):
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "Pad":
            mode = "constant"
            pads = []
            value = 0.0
            for a in node.attribute:
                if a.name == "pads":
                    pads = a.data
                    # print("pad:", pads)
                if a.name == "mode":
                    mode = bytes.decode(a.data[0], encoding="utf-8")
                    # print("mode:", mode)
                if a.name == "value":
                    value = a.data[0]
                    # print("value:", value)
            
            if len(pads) != 8:
                logger.warn("Pads attribute len not 8.")
                return False
            
            temp_pad = [pads[0], pads[1], pads[4], pads[5]]
            for i in temp_pad:
                if i != 0:
                    logger.warn("Pads value invalid.")
                    return False

            if (value == 0.0 and mode == "constant"):
                if node.next_node[0].op_type == "Conv":
                    return True
        
        return False


    # 运行一次优化
    def run_pass(self, ir_graph):
        for node in ir_graph.node_list:
            if self.match_conditions(node):
                logger.info("---- fuse pad into Conv. %s", node.output[0].name)

                if (len(node.next_node[0].input) != 1):
                    logger.error("error. next node has multi input.")
                    sys.exit(-1)

                # 修改下层node的pads属性
                pads_1 = []
                pads_2 = []
                for a in node.attribute:
                    if a.name == "pads":
                        pads_1 = a.data
                if len(pads_1) == 0: # todo . convert auto_pad to pads
                    logger.error("not support auto pad. %s", pads_1)
                    sys.exit(-1)

                for a in node.next_node[0].attribute:
                    if a.name == "pads":
                        pads_2 = a.data

                        if (len(pads_2) != 4):
                            logger.error("Pads attribute len not 4.")
                            sys.exit(-1)

                        pads_2[0] += pads_1[2]
                        pads_2[1] += pads_1[3]
                        pads_2[2] += pads_1[6]
                        pads_2[3] += pads_1[7]

                        a.data = pads_2
                        logger.info("Conv pads = %s", a.data)

                # 删除当前node
                node.next_node[0].input = node.input
                ir_graph.node_list.remove(node)
                return True
                
        return False