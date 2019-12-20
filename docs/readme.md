# YKX ONNX Operator Converter for ONNC compiler

## ONNC 支持的 ONNX operator
- Add
- AveragePool
- BatchNormalization
- Concat
- Conv
- Gemm
- GlobalAveragePool
- LRN
- MaxPool
- Mul
- Relu
- Reshape
- Softmax
- Sum
- Unsqueeze
- Transpose (In ShuffeNet case)
---

## ONNX OP converter的一般性假设
1. 所有input tensor的尺寸都是固定的，且排列方式为：[n, c, h, w]
2. 所有的ONNX model都会经过shape inference，即对于任意一个node，input tensor和output tensor是固定且已知的
3. 转换后的ONNX model仅用于inference，不会重新导入到框架中进行train
4. ONNX model中的所有属性均合法

### 1. Dropout remove
![dropout remove](./picture/graphviz/001_dropout_remove/dropout_remove.png)  
##### 类型：
假指令移除
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)
Dropout在inference中无作用，所以可以被移除
##### Attributes 限制：
无
##### 受影响的框架：
|  框架   | 影响  |
|  ----  | :----:  |
| Caffe  | 是 |
| Caffe2  | 是 |
| PyTorch  | 是 |
| Keras | 是 |
| TensorFlow | 是 |
| MXNet | 是 |

---

### 2. Flatten -> Reshape
![flatten -> reshape](./picture/graphviz/002_flatten_to_reshape/flatten_to_reshape_0.png)
##### 类型：
等效指令替换
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten)
Flatten：将n维的input tensor转换成2维output tensor（也就是matrix）  
Reshape：在元素数量相同的条件下，将n维的input tensor转换成m维的output tensor    
所以说flatten是reshape的一种特例。
##### Attributes 限制：
无
##### 转换公式：
```
input_tensor_dim := n  
input_tensor_shape := [d_0, d_1, ... , d_n-1]  
Flatten->Axis := a  
if a ==0 :   
Reshape->Shape := [1, d_0\*d_1\* ... \*dn_1]  
else :   
Reshape->Shape := [d_0\*...*d_a-1, d_a\*...\*d_n-1]  
```
##### 常用特例：
Flatten和Reshape一般都是用在fc layer之前用于将tensor转换成vector:  
```
Flatten->Axis = 0  
Reshape->Shape = [1, n\*c\*h\*w]  
```
![flatten -> reshape](./picture/graphviz/002_flatten_to_reshape/flatten_to_reshape_1.png)
##### 受影响的框架：
|  框架   | 影响  |
|  ----  | :----:  |
| Caffe  | 是 |
| Caffe2  | 是 |
| PyTorch  | 是 |
| Keras | 是 |
| TensorFlow | 是 |
| MXNet | 是 |

---

### 3. Constant  remove
![Constant remove](./picture/graphviz/003_constant_remove/1.png)  
##### 类型：
假指令移除
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)
用node表示一个静态张量，张量数据存放于Attributes中，可以将该张量放入模型的初始化weight中，然后删除该node。
##### Attributes 限制：
无


---

### 4. Identity  remove
![Constant remove](./picture/graphviz/004_identity_remove/1.png)  
##### 类型：
假指令移除
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity)
拷贝输入到输出，仅是复制tensor操作，可以直接删除
##### Attributes 限制：
无

---

### 5. Shape  remove
![Shape remove](./picture/graphviz/005_shape_remove/1.png)  
##### 类型：
假指令移除
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape)
输出input tensor的形状。因为假定graph的input输入形状是静态的，所以对于中间Node的input输入形状也是固定的，所以可以把该Shape的输出保存到初始化weight中,然后删除该Node.
##### Attributes 限制：
无

---


### 6. Pad remove
![Pad remove](./picture/graphviz/006_pad_remove/1.png)  
##### 类型：
假指令移除

##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)
对于无效的pad 操作可以直接删除。
从pytorch,tensorflow等框架导出的onnx model 经常会带有无效的pad 

##### Attributes 限制：
pads值全为0  或者

---


### 6. Pad fuse into Pooling 
![Pad fuse](./picture/graphviz/006_pad_remove/2.png)  
##### 类型：
指令融合
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)
对于 pad + pooling 这样的操作，由于pooling操作有有pads参数可是设置，所以可以将一些pad层与pooling层融合
##### Attributes 限制：
mode="constant" , constant_value=0, pads值不小于0

---



### 6. Pad fuse into Conv 
![Pad fuse](./picture/graphviz/006_pad_remove/2.png)  
##### 类型：
指令融合
##### 原理：
[Operator Define](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)
对于 Pad + Conv 这样的操作，由于Conv操作有有pads参数可是设置，所以可以将一些Pad层与Vonc层融合
##### Attributes 限制：
mode="constant" , constant_value=0, pads值不小于0

---