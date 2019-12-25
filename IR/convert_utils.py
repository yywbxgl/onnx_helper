import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
import logging
import struct # https://docs.python.org/3/library/struct.html


logging.basicConfig(level=logging.DEBUG)

from IR import ir

# input_type = IR.Value()
def convert_raw_data(value_data):
    if value_data.raw == False or len(value_data.data) == 0:
        logging.info("can not convert raw data")
        return value_data

    format_str = ""
    if value_data.data_type == ir.DataType.FLOAT.value : 
        format_str = "<f"
    elif value_data.data_type == ir.DataType.INT8.value:
        format_str = "<c"
    elif value_data.data_type == ir.DataType.INT16.value:
        format_str = "<h"
    elif value_data.data_type == ir.DataType.INT32.value:
        format_str = "<i"
    elif value_data.data_type == ir.DataType.INT64.value:
        format_str = "<q"
    elif value_data.data_type == ir.DataType.DOUBLE.value:
        format_str = "<d"
    else:
        logging.error("can not convert raw data. data_type:", value_data.data_type)

    if format_str != "":
        data_temp = struct.iter_unpack(format_str, bytes(value_data.data))
        new_data =  [i[0] for i in data_temp]
        print(new_data)
        value_data.data = new_data
        value_data.raw = False

    return value_data