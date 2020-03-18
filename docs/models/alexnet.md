# AlexNet
decription xxxx

## Download:  
producer | downloada | ir version |  opset | framework |
--- | --- | --- | --- | --- |
onnx model zoo |[bvlc_alexnet.onnx](ftp://172.16.1.15/%CA%FD%BE%DD%BC%AF%D3%EB%B1%EA%D7%BC%C4%A3%D0%CD%BF%E2/ykx%20model%20zoo/Image%20Classification/001_AlexNet/bvlc_alexnet.onnx)  | 3 | 8 | n/a |
pytorch        |[alexnet-pytorch.onnx](超链接xxx)  | 4 | 9 | n/a |
tensorflow     |[alexnet-tf.onnx](超链接xxx)  | 5 | 7 | n/a |
keras          |[searching…](超链接xxx)  |  |  |  |


## Convert: 
origin model | op converted model | version converted model |  loadable |
--- | --- | --- | --- | 
[alexnet-pytorch.onnx](超链接xxx) | [alexnet-pytorch_optimized.onnx](超链接xxx) |  [alexnet-pytorch_optimized_ir3.onnx](超链接xxx) | [alexnet-pytorch_optimized_ir3.nbdla](超链接xxx) |



## Results/accuracy on test set
model | top1-onnx | top1-loadable |  top1-diff | top5-onnx | top5-loadable| top5-diff|
--- | --- | --- | --- | --- | ---| ---|
[bvlc_alexnet.onnx](ftp://172.16.1.15/%CA%FD%BE%DD%BC%AF%D3%EB%B1%EA%D7%BC%C4%A3%D0%CD%BF%E2/ykx%20model%20zoo/Image%20Classification/001_AlexNet/bvlc_alexnet.onnx) | 54.498% | 53.858% | -0.64% | 78.15% | 77.516% | -0.63% |
[alexnet-pytorch.onnx](超链接xxx) |


## Model input and output
- input: float[1, 3, 224, 224]  
- output: float[1, 1000]


## Pre-processing steps
- bgr
- mean 103.939 ,116.779,123.68

## Post-processing steps


## Dataset
[ILSVRC2012](链接地址xxxx)



