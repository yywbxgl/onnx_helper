import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from graphviz import Digraph
import logging
logger = logging.getLogger(__name__)

from checker import operator_list
from IR import pb_to_ir

def ir_op_check(ir_graph):

    ret = True
    g = Digraph('G', filename='test', format='png')
    g.attr(rankdir='TB')
    g.attr('node', shape='box', color="red", style='filled')
    logger.info("check graph operator...")
    for node in ir_graph.node_list:
        for i in node.input:
            if i.name in [t.name for t in ir_graph.input] :
                for t in ir_graph.input:
                    if t.name == i.name:
                        g.edge(node.name, t.name, label=str(t.dims))
        for i in node.output:
            if i.name in [t.name for t in ir_graph.output] :
                for t in ir_graph.output:
                    if t.name == i.name:
                        g.edge(node.name, t.name, label=str(t.dims))
        for node2 in node.next_node:
            for out in node.output:
                if out in node2.input:
                    g.edge(node.name, node2.name, label=str(out.dims))
        if node.op_type in operator_list.skym_operator_list:
            g.node(node.name, color="green", style='filled')
            logger.debug(" %s ", node.op_type)
        elif node.op_type in operator_list.nbdla_operator_list:
            g.node(node.name, color="yellow", style='filled')
            logger.warn(" %s not support. ", node.op_type)
            ret = False
        else:
            logger.warn(" %s not support. ", node.op_type)
            ret = False

    g.node("Input", color="green", style='filled')
    g.node("Output", color="green", style='filled')

    # g.save()
    # g.view()
    return ret

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], "onnx_model_file")
        sys.exit(-1)

    graph = pb_to_ir.convert(sys.argv[1])
    graph.dump()

    ir_op_check(graph)




