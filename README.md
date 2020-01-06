## Introduction
- ##### onnx model optimize optimizer
  - Modify onnx operator, 
  - optimize the graph.
  - See details in [YKX ONNX Operator Converter.](/docs/Operator_Converter.md)

- ##### onnx model editor
   - export onnx model config and weight, 
   - modity the model by config, 


## Environment
pip3 install onnx  
pip3 install graphviz


## Example

```
cd example

# create model graph picture
python3  dot_test.py   alexnet.onnx

# optimize the onnx model
python3  optimizer_test.py  alexnet.onnx   alexnet_out

# export onnx model config and weight 
python3  export_onnx_model.py   OnnxModel   out_path

# create onnx model by config and weight
python3  create_onnx_model.py  config_file  weights_path
```