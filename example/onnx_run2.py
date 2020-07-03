import numpy as np
import onnx
import onnxruntime
import onnxruntime.backend as backend
#import caffe2.python.onnx.backend as backend
import os,sys

import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)-5s] [%(filename)s:%(lineno)d] %(message)s"
# coloredlogs.install(level="INFO", fmt=fmt)
coloredlogs.install(level="DEBUG", fmt=fmt)
logger = logging.getLogger(__name__)


if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], " OnnxModel")
    sys.exit(-1)

model = onnx.load(sys.argv[1])
session = backend.prepare(model)
onnx.checker.check_model(model)

x = np.random.randn(1, 3, 48, 48).astype(np.float32) 
output = session.run(x)

for i in output:
    print(np.array(i).shape)

print(output)


# print(onnxruntime.get_device())

# soption = onnxruntime.SessionOptions()
# print(soption.graph_optimization_level)
# soption.log_severity_level = 0
# soption.log_verbosity_level = 0
# soption.optimized_model_filepath = "./11.onnx"
# sess = onnxruntime.InferenceSession(sys.argv[1],sess_options=soption, providers=[])


# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name
# print(input_name)
# print(output_name)


# roption = onnxruntime.RunOptions()
# roption.log_severity_level = 0
# roption.log_verbosity_level = 0
# output = sess.run([output_name], {input_name: x.astype(np.float32)}, run_options = roption)






