import numpy as np
import onnx
import onnxruntime
import onnxruntime.backend as backend
#import caffe2.python.onnx.backend as backend
import os,sys



if len(sys.argv) != 3:
    print ("Usage:", sys.argv[0], " OnnxModel—1  OnnxModel-2")
    sys.exit(-1)


model1 = onnx.load(sys.argv[1])
session1 = backend.prepare(model1)

model2 = onnx.load(sys.argv[2])
session2 = backend.prepare(model2)


# get input size
input_size = []
for i in model1.graph.input[0].type.tensor_type.shape.dim:
    input_size.append(i.dim_value)
print("input_name:", model1.graph.input[0].name)
print("input_size:", input_size)


# get input data
x = np.random.random(input_size).astype(np.float32)*255


# Run the model on the backend
print("onnx run 1 ...")
output1 = session1.run(x)
print("output1:", len(output1))
print("onnx run 2 ...")
output2 = session2.run(x)
print("output2:",len(output2))

for i in range(len(output1)):   
    np.testing.assert_allclose(output1[i], output2[i], rtol=1e-3, atol=1e-4)
print("test ok")



