import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import onnx

from IR import ir
from IR import pb_to_ir
from IR import ir_to_pb
from creator import ir_to_config
from creator import config_to_ir

# 导出一个onnx model的配置文件以及全部参数
def exportModel(onnx_model, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # convert model to ir_graph
    ir_graph = pb_to_ir.convert(onnx_model)

    # save config file
    print("export config ...")
    config = ir_to_config.exportConfig(ir_graph)
    file_name = os.path.join(save_dir, "config.txt")
    ir_to_config.save_config(config, file_name)

    # save weight
    print("export weight ...")
    for node in ir_graph.node_list:
        for w in node.weight:
            if w.raw == False:
                data = np.array(w.data)
                file_name = os.path.join(save_dir, w.name)
                np.save(file_name, data)
            else:
                data = np.array(w.data).astype(np.byte)
                file_name = os.path.join(save_dir, w.name)
                np.save(file_name, data)
                
    print("export %s to %s success."%(onnx_model, save_dir))


# 通过配置文件以及参数文件生成一个onnx model
# 如果参数文件不存在，则参数值根据dims随机填充
def createModel(weight_dir, output_mode):
    files= os.listdir(weight_dir)
    config = os.path.join(weight_dir, 'config.txt')
    ir_graph = config_to_ir.importConfig(config)

    # 从npy文件初始化weight
    for node in ir_graph.node_list:
        for w in node.weight:
            if w.name +".npy" in files:
                print("copy weight data from npy ", w.name)
                data = np.load(os.path.join(weight_dir, w.name +".npy"))
                w.data = data
                # w.data = list(data.flatten())
                if type(data[0]) == np.byte:
                    print("set raw")
                    w.raw = True

                # PS: reshape等operator的weight的数据类型为INT
                if (data.dtype == np.int64 or data.dtype == np.int32 or data.dtype == np.int16):
                    w.data_type = 7

    model = ir_to_pb.convert(ir_graph)
    file_name = output_mode
    onnx.save(model, file_name)

    print("save model %s success"%(file_name))

    return ir_graph


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " OnnxModel")
        sys.exit(-1)

    # export model
    exportModel(sys.argv[1], "out")
    
    # create model
    createModel("out/CNTKGraph_cfg_2.txt", "./")

