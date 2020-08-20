import onnx
from onnx import version_converter, helper

# Preprocessing: load the model to be converted.
model_path = '/home/sunqiliang/share/yolov3-tiny-ykx.onnx'
original_model = onnx.load(model_path)

print('The model before conversion:', original_model.ir_version, original_model.opset_import[0])

# A full list of supported adapters can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/version_converter.py#L21
# Apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, 8)
converted_model.ir_version = 3

print("save model.")
onnx.save(converted_model, "11.onnx")

