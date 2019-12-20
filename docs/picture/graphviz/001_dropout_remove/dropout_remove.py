from graphviz import Digraph

g = Digraph('G', filename='dropout_remove', format='png')

# # NOTE: the subgraph name needs to begin with 'cluster' (all lowercase)
# #       so that Graphviz recognizes it as a special cluster subgraph

g.attr(rankdir='TB')

# with g.subgraph(name='cluster_1') as c:
#     c.attr(rankdir='LR')
#     c.attr(color='black')
#     c.node('pdp_0', style='filled')
#     c.node('pdp_1')
#     c.attr(label='PDP')

# with g.subgraph(name='cluster_2') as c:
#     c.attr(rankdir='LR')
#     c.attr(color='blue')
#     c.node('pdp_1_output')
#     c.attr(label='DRAM')

# g.edge('pdp_0_input', 'pdp_0')
# g.edge('pdp_0_output', 'pdp_1')
# g.edge('pdp_1', 'pdp_1_output')
# g.edge('pdp_0', 'pdp_0_output')

# index = 1
with g.subgraph(name='1') as c:
    c.attr(rankdir='TB')
    # c.attr(color='black')
    c.node('src_node_0', 'src node')
    c.node('dropout', shape='Mrecord', label='''<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0" TITLE='index'>
        <tr>
            <td BGCOLOR="black" COLSPAN="2" WIDTH="120"><FONT COLOR="white"><B>dropout</B></FONT></td>
        </tr>
        <tr>
            <td ALIGN="LEFT" WIDTH="60"></td>
            <td WIDTH="60"></td>
        </tr>
    </TABLE>>''')
    c.node('dst_node_0', 'dst node')
    c.edge('src_node_0', 'dropout', label='\\ \\ [n, c, h, w]')
    c.edge('dropout', 'dst_node_0', label='\\ \\ [n, c, h, w]')

with g.subgraph(name='2') as c:
    c.attr(rankdir='TB')
    # c.attr(color='black')
    c.node('src_node_1', 'src node')
    c.node('dst_node_1', 'dst node')
    c.edge('src_node_1', 'dst_node_1', label='\\ \\ [n, c, h, w]')

g.body.append('\t{rank=same;"src_node_0"; "src_node_1"}\t')
g.body.append('\t{rank=same;"dst_node_0"; "dst_node_1"}\t')
#print(g)

g.view()