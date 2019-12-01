import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
from graphviz import Digraph

from checker import operator_list
from IR import pb_to_ir

def ir_dot(ir_graph):
    g = Digraph('G', filename='test', format='png')
    g.attr(rankdir='TB')
    g.attr('node', shape='box')
    for node in ir_graph.node_list:
        for node2 in node.next_node:
            g.edge(node.name, node2.name)
    g.view()


def ir_dot2(ir_graph):
    g = Digraph('G', filename='test2', format='png')
    g.attr(rankdir='TB')
    g.attr('node', shape='box', color="red", style='filled')
    for node in ir_graph.node_list:
        for i in node.input:
            if i.name == ir_graph.input.name:
                g.edge("Input", node.name, label=str(ir_graph.input.dims))
        for i in node.output:
            if i.name == ir_graph.output.name:
                g.edge(node.name, "Output", label=str(ir_graph.output.dims))
        for node2 in node.next_node:
            for out in node.output:
                if out in node2.input:
                    g.edge(node.name, node2.name, label=str(out.dims))
        if node.op_type in operator_list.skym_operator_list:
            g.node(node.name, color="green", style='filled')
        if node.op_type in operator_list.nbdla_operator_list:
            g.node(node.name, color="yellow", style='filled')
    g.node("Input", color="green", style='filled')
    g.node("Output", color="green", style='filled')

    # g.save()
    g.view()

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], "onnx_model_file")
        sys.exit(-1)

    graph = pb_to_ir.convert(sys.argv[1])
    pb_to_ir.dump(graph)

    # ir_dot(graph)
    ir_dot2(graph)




