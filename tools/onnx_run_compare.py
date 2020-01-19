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


# get input data
x = np.random.randn(1, 3, 224, 224).astype(np.float32) 
# print(x)

# Run the model on the backend
print("onnx run 1 ...")
output1 = session1.run(x)
print("onnx run 2 ...")
output2 = session2.run(x)


np.testing.assert_allclose(output1, output2, rtol=1e-3, atol=1e-4)
print("test ok")

compare = np.array(output1)-  np.array(output2)
print("inference result compare diff: ", compare.sum())
if compare.sum() > 0.1:
    print("inference not pass!!!")
else:
    print("test ok")
    
    

# print(output1)
# print(output2)


