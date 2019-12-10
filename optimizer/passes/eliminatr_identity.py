import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

from optimizer import eliminate_node

def run(graph):
    return  eliminate_node.run(graph, "Identity")



