import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import onnx
import numpy as np
import onnxruntime.backend as backend
import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)-5s] [%(filename)s:%(lineno)d] %(message)s"
coloredlogs.install(level="INFO", fmt=fmt)
# coloredlogs.install(level="DEBUG", fmt=fmt)
logger = logging.getLogger(__name__)

from IR import pb_to_ir
from IR import ir_to_pb
from optimizer import operator_convert

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "onnx_model  output_model")
        sys.exit(-1)

    # pb_to_ir
    graph = pb_to_ir.convert(sys.argv[1])
    graph.dump()

    # optimize
    graph = operator_convert.run_all_pass(graph)
    # graph.dump()
  
    # pb_to_ir
    onnx_model = ir_to_pb.convert(graph)
    logger.info('save onnx model ...')
    onnx.save(onnx_model, sys.argv[2]+".onnx")

    # check model, inference compare
    logger.info("inference compare...")
    input_shape = graph.input.dims
    input_data = np.random.randint(0,255, size=input_shape).astype(np.float32)
    logger.debug("input_shape:%s", str(input_shape))

    model_1 = onnx.load(sys.argv[1])
    session_1 = backend.prepare(model_1,  strict=False)
    output_1 = session_1.run(input_data)
    output_1 = np.array(output_1)
    logger.info("input model test finish")

    model_2 = onnx.load(sys.argv[2]+".onnx")
    session_2 = backend.prepare(model_2,  strict=False)
    output_2 = session_2.run(input_data)
    output_2 = np.array(output_2)
    logger.info("output model test finish")

    compare = output_1 - output_2

    logger.info("inference result compare diff: %f", compare.sum())
    if compare.sum() > 1:
        logging.error("inference not pass!!!")
    else:
        logger.info("test ok")

