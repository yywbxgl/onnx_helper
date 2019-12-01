class Value:
    def __init__(self):
        self.name = ""          # weight名称
        self.data_type=0        # weight值类型, 枚举表示，具体值见 onnx.proto, 
        self.dims = []          # weigph形状
        self.data = []          # weight值
        self.raw = False        # data是否二进制存储，type(slef.data) = bytes


class Node:
    def __init__(self):
        self.name = ""       # 可以不填
        self.op_type = ""    # operator名称
        self.attribute = []  # operator相关参数, type:Value
        self.pre_node = []   # 指向上一层Node
        self.next_node = []  # 指向下一层Node
        self.weight = []     # 保存当前Node的参数， type:Value
        self.input = []      # 指向pre_node.output
        self.output = []     # 唯一名称，可能有多个output, type:Value
        # self.rank          # 用于表示node 之间的运行顺序 


class Graph:
    def __init__(self):
        self.name = ""          # graph 的名称
        self.node_list = []     # graph 的网络结构, type:Node
        self.input = Value()    # graph 的输入层信息，type:value  只支持一个input
        self.output = Value()   # graph 的输出层信息，type:value  只支持一个output
        # self.init_dict = {}          # 存放weight数据
        # self.mid_feature_dict = {}   # 存放中间层信息

