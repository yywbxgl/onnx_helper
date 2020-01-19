#!/bin/sh -v

echo "convert model to ir_version 3：$1";
rm  ../example/out/*


sudo pip3 install onnx==1.4.1 --upgrade
python3 ../example/export_onnx_model.py   $1   ../example/out


sudo pip3 install onnx==1.3.0 --upgrade
python3 ../example/create_onnx_model.py  ../example/out/  $1_ir3.onnx


sudo pip3 install onnx==1.4.1 --upgrade
python3  onnx_run_compare.py  $1  $1_ir3.onnx


./onnc.nv_large.151   $1_ir3.onnx  -o  $1_ir3.nbdla
