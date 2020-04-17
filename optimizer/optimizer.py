import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
logger = logging.getLogger(__name__)

from checker import operator_list


class Optimizer():

    def __init__(self):
        # 注册passCase到Manager
        from optimizer.passes.convert_concat_to_init import convert_concat_to_init
        from optimizer.passes.convert_constant_to_init import convert_constant_to_init
        from optimizer.passes.convert_flatten_to_reshape import convert_flatten_to_reshape
        from optimizer.passes.convert_gather_to_init import convert_gather_to_init
        from optimizer.passes.convert_reduceMean_to_globalAveragePool import convert_reduceMean_to_globalAveragePool
        from optimizer.passes.convert_shape_to_init import convert_shape_to_init
        from optimizer.passes.convert_unsuqeeze_to_init import convert_unsuqeeze_to_init

        from optimizer.passes.eliminate_dropout import eliminate_dropout
        from optimizer.passes.eliminate_identity import eliminate_identity
        from optimizer.passes.eliminate_pad import eliminate_pad

        from optimizer.passes.fuse_pad_into_averagePool import fuse_pad_into_averagePool
        from optimizer.passes.fuse_pad_into_conv import fuse_pad_into_conv
        from optimizer.passes.fuse_pad_into_maxPool import fuse_pad_into_maxPool

        self.passes_manager= {}
        
        self.passes_manager["convert_concat_to_init"] = convert_concat_to_init()
        self.passes_manager["convert_constant_to_init"] = convert_constant_to_init()
        self.passes_manager["convert_flatten_to_reshape"] = convert_flatten_to_reshape()
        self.passes_manager["convert_gather_to_init"] = convert_gather_to_init()
        self.passes_manager["convert_reduceMean_to_globalAveragePool"] = convert_reduceMean_to_globalAveragePool()
        self.passes_manager["convert_shape_to_init"] = convert_shape_to_init()
        self.passes_manager["convert_unsuqeeze_to_init"] = convert_unsuqeeze_to_init()

        self.passes_manager["eliminate_dropout"] = eliminate_dropout()
        self.passes_manager["eliminate_identity"] = eliminate_identity()
        self.passes_manager["eliminate_pad"] = eliminate_pad()

        self.passes_manager["fuse_pad_into_averagePool"] = fuse_pad_into_averagePool()
        self.passes_manager["fuse_pad_into_conv"] = fuse_pad_into_conv()
        self.passes_manager["fuse_pad_into_maxPool"] = fuse_pad_into_maxPool()


    # 获取当前支持的optimize选项
    def get_all_optimize(self):
        ret_list = []
        for i in self.passes_manager:
            ret_list.append(i)
        return ret_list
            
    # 检测graph满足的optimize选项
    def check_graph(self, graph):
        ret_list = []
        for node in graph.node_list:
            for case in self.passes_manager:
                if self.passes_manager[case].match_conditions(node) == True and case not in ret_list:
                    ret_list.append(case)
        return ret_list

    # 循环执行单个optimize,直到整个graph
    def run(self, graph, pass_name):
        optimized_flag = False
        while 1:
            if self.passes_manager[pass_name].run_pass(graph) == True:
                optimized_flag = True
            else:
                break
        return optimized_flag

    # 对整个grapg进行optimize
    def optimize_graph(self, graph, pass_list):
        for i in pass_list:
            if i not in self.passes_manager:
                logger.error("not support opmitize: %s", i)
                sys.exit(-1)
        
        while 1:
            continue_flag = False
            for i in pass_list:
                logger.debug("optimize_graph: %s", i)
                ret = self.run(graph, i)
                if ret == True:
                    continue_flag = True

            if continue_flag == False:
                break
        
        return graph

class PassCase():

    # 判断node满足optimize条件
    def match_conditions(self, node):
        logging.warn("no implement.")
        return False

    # 运行一次optimize
    def run_pass(self, graph):
        logging.warn("no implement.")
        return False



if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnx_model  output_model")
        sys.exit(-1)

    import onnx
    from IR import pb_to_ir
    from IR import ir_to_pb

    # pb_to_ir
    graph = pb_to_ir.convert(sys.argv[1])
    graph.dump()

    # optimize the grapg
    pass_list = ["eliminate_dropout"]
    graph = Optimizer().optimize_graph(graph, pass_list)
  
    # ir_to_pb
    onnx_model = ir_to_pb.convert(graph)
    print('save onnx model ...')
    onnx.save(onnx_model, sys.argv[2]+".onnx")
    