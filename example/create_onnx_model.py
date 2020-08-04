import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from creator import ir_creator

import logging
import coloredlogs
fmt = "[%(levelname)-5s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
fmt = "[%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
# fmt = "%(filename)s:%(lineno)d %(levelname)s - %(message)s"
coloredlogs.install(level="DEBUG", fmt=fmt)
# coloredlogs.install(level="INFO", fmt=fmt)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "config_path  output_model")
        sys.exit(-1)
    
    # create model
    ir_creator.createModel(sys.argv[1], sys.argv[2])
