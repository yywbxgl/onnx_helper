import numpy as np
import onnx
import onnxruntime
import os,sys


if len(sys.argv) != 2:
    print ("Usage:", sys.argv[0], " OnnxModel")
    sys.exit(-1)


session = onnxruntime.InferenceSession(sys.argv[1])

inputs = session.get_inputs()
outputs = session.get_outputs()


print("-----inputs----------")
inputs_dict = {}
for i in inputs:
    print(i.name, i.shape, i.type)
    x = np.random.sample(i.shape).astype(np.float32) * 255
    inputs_dict[i.name] = x


print("-----outputs----------")
output_list = []
for i in outputs:
    print(i.name, i.shape, i.type)
    output_list.append(i.name)


print("-----inference----------")
result = session.run(output_list, inputs_dict)
print("test pass.")
# print(result)







