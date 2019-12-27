from enum import Enum

class Value:
    def __init__(self):
        self.name = ""          # weight名称
        self.data_type=0        # weight值类型, 枚举表示，具体值见 onnx.proto, 
        self.dims = []          # weigph形状
        self.data = []          # weight值
        self.raw = False        # data是否二进制存储，type(slef.data) = bytes
        self.init = False       # 是否有初始值


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

    def dump(self):
        print("-------- ir_grapg dump --------------")
        print("graph_input  =", self.input.name,  self.input.dims)
        print("graph_output =", self.output.name,  self.output.dims)
        for i in self.node_list:
            print("node = ", i.name) 
            for inp in i.input:
                print("\tinput:", inp.name, inp.dims)
            for out in i.output:
                print("\toutput:", out.name, out.dims)
            for attr in i.attribute:
                print("\tattr:", attr.name, attr.data)
            for w in i.weight:
                print("\tweight:", w.name, w.dims)
            for p in i.pre_node:
                print("\tpre_node:", p.name)
            for n in i.next_node:
                print("\tnext_node:", n.name)
        print("-----------------------------------")

    # 更新graph的名称以及pre_node和next_node
    def updata_graph(self):
        # ----- 初始化-------
        for node in self.node_list:
            node.pre_node = []
            node.next_node = []

        # ----- 遍历graph，填充pre_node 与 next_node -----
        for node in self.node_list:
            # print("node[%s] to find pre_node"%(node.name))
            for i in node.input:
                for node2 in self.node_list:
                    if i in node2.output:
                        node.pre_node.append(node2)
                        # print(node2.name)

            # print("node[%s] to find next_node"%(node.name))
            for o in node.output:
                for node2 in self.node_list:
                    if o in node2.input:
                        node.next_node.append(node2)
                        # print(node2.name)

        # ----- 重命名Node.name, 确保名称唯一------
        print("---------------------")
        print("rename node name ...")
        for index, node in  enumerate(self.node_list):
            node.name = node.op_type + "_" + str(index)


# 用于表示weight 以及 input, mid freature 等tensor的数据类型
# PS: Node.attribute.data_type 不适用该枚举  todo
class DataType(Enum):
    UNDEFINED = 0
    FLOAT = 1  
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9 

    # IEEE754 half-precision floating-point format (16 bits wide).
    # This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10

    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14    # complex with float32 real and imaginary components
    COMPLEX128 = 15   # complex with float64 real and imaginary components

    # Non-IEEE floating-point format based on IEEE754 single-precision
    # floating-point number truncated to 16 bits.
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16


    
if __name__ == "__main__":
    print(DataType.FLOAT.value)
    print(DataType["FLOAT"].value)
    print(DataType(1).name)

    test = DataType.FLOAT
    print(test)
    print(test.value)
    print(test.name)

    
    


   

