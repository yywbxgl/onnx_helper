import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
import struct # https://docs.python.org/3/library/struct.html
import logging
logger = logging.getLogger(__name__)

from IR import ir

# input_type = IR.Value()
def convert_raw_data(value_data):
    if value_data.raw == False or len(value_data.data) == 0:
        # logging.info("can not convert raw data")
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
        logger.error("can not convert raw data. data_type: %s", value_data.data_type)

    if format_str != "":
        data_temp = struct.iter_unpack(format_str, bytes(value_data.data))
        new_data =  [i[0] for i in data_temp]
        # print(new_data)
        value_data.data = new_data
        value_data.raw = False
        logger.debug("convert raw data success.")

    return value_data



# input_type = IR.Value()
def get_raw_data(value_data):

    ret = value_data.data
    
    # 转换raw_data
    if value_data.raw == True:
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
            logger.error("can not convert raw data. data_type: %s", value_data.data_type)

        if format_str != "":
            data_temp = struct.iter_unpack(format_str, bytes(value_data.data))
            ret =  [i[0] for i in data_temp]
            # print(ret)
            logger.debug("convert raw data success.")
    
    # 转换标量
    if  value_data.dims == []:
        if type(ret) == type([]):
            ret = ret[0]

    return ret

