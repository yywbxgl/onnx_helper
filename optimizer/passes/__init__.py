import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import eliminate_dropout
import eliminate_identity
import eliminate_pad
import convert_flatten_to_reshape
import convert_shape_to_init
import convert_constant_to_init
import fuse_pad_into_averagePooling