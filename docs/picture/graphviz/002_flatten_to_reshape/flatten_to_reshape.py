from graphviz import Digraph
g = Digraph('G', filename='flatten_to_reshape_0', format='png')

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
    c.node('flatten', shape='Mrecord', label='''<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0" TITLE='index'>
        <tr>
            <td BGCOLOR="black" COLSPAN="2" WIDTH="120"><FONT COLOR="white"><B>flatten</B></FONT></td>
        </tr>
        <tr>
            <td ALIGN="LEFT" WIDTH="60">Axis:</td>
            <td WIDTH="60">a</td>
        </tr>
    </TABLE>>''')
    c.node('dst_node_0', 'dst node')
    c.edge('src_node_0', 'flatten', label='\\ \\ [d_0, d_1, ... , d_n-1]')
    c.edge('flatten', 'dst_node_0', label='\\ \\ [d_0&times;...&times;d_a-1, d_a&times;...&times;d_n-1]')

with g.subgraph(name='2') as c:
    c.attr(rankdir='TB')
    # c.attr(color='black')
    c.node('src_node_1', 'src node')
    c.node('reshape', shape='Mrecord', label='''<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0" TITLE='index'>
        <tr>
            <td BGCOLOR="black" COLSPAN="2" WIDTH="100"><FONT COLOR="white"><B>reshape</B></FONT></td>
        </tr>
        <tr>
            <td ALIGN="LEFT" WIDTH="50">shape:</td>
            <td WIDTH="50">[d_0&times;...&times;d_a-1, d_a&times;...&times;d_n-1]</td>
        </tr>
    </TABLE>>''')
    c.node('dst_node_1', 'dst node')
    c.edge('src_node_1', 'reshape', label='\\ \\ [d_0, d_1, ... , d_n-1]')
    c.edge('reshape', 'dst_node_1', label='\\ \\ [d_0&times;...&times;d_a-1, d_a&times;...&times;d_n-1]')

#print(g)

g.view()


g = Digraph('G', filename='flatten_to_reshape_1', format='png')

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
    c.node('flatten', shape='Mrecord', label='''<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0" TITLE='index'>
        <tr>
            <td BGCOLOR="black" COLSPAN="2" WIDTH="120"><FONT COLOR="white"><B>flatten</B></FONT></td>
        </tr>
        <tr>
            <td ALIGN="LEFT" WIDTH="60">Axis:</td>
            <td WIDTH="60">0</td>
        </tr>
    </TABLE>>''')
    c.node('dst_node_0', 'dst node')
    c.edge('src_node_0', 'flatten', label='\\ \\ [n, c, h, w]')
    c.edge('flatten', 'dst_node_0', label='\\ \\ [1, n&times;c&times;h&times;w]')

with g.subgraph(name='2') as c:
    c.attr(rankdir='TB')
    # c.attr(color='black')
    c.node('src_node_1', 'src node')
    c.node('reshape', shape='Mrecord', label='''<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0" TITLE='index'>
        <tr>
            <td BGCOLOR="black" COLSPAN="2" WIDTH="100"><FONT COLOR="white"><B>reshape</B></FONT></td>
        </tr>
        <tr>
            <td ALIGN="LEFT" WIDTH="50">shape:</td>
            <td WIDTH="50">[1, n&times;c&times;h&times;w]</td>
        </tr>
    </TABLE>>''')
    c.node('dst_node_1', 'dst node')
    c.edge('src_node_1', 'reshape', label='\\ \\ [n, c, h, w]')
    c.edge('reshape', 'dst_node_1', label='\\ \\ [1, n&times;c&times;h&times;w]')

#print(g)

g.view()