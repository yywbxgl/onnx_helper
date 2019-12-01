import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from optimizer import eliminate_node

def run_case(graph):
    graph = eliminate_node.run(graph, "Dropout")
    return graph



