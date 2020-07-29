import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
import logging
logger = logging.getLogger(__name__)

from IR import ir


class matmul_to_gemm():
        
    # 判断是否满足条件
    def match_conditions(self, node):
        if node.op_type == "MatMul":
            return True

        return False


    def run_pass(self, graph):  
        for node in graph.node_list:
            if self.match_conditions(node) == True:

                # input  不需要改变， attribute 均使用默认值                 
                node.name = ""
                node.op_type = "Gemm"
                node.attribute = []  

                # 添加 weight C.  bias 填0
                new_weight = ir.Value()
                new_weight.name =  node.output[0].name + "_Gemm_C"
                new_weight.dims = [1]
                new_weight.data = [0.0]
                new_weight.data_type = 1 # FLOAT32

                node.weight.append(new_weight)

            
                logger.info("---- convert  MatMul to Gemm , node:%s", node.output[0].name)
                return True

        return False

