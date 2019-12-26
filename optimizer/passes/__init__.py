import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import eliminate_dropout
import eliminate_identity
import eliminate_pad
import convert_flatten_to_reshape
import convert_shape_to_init
import convert_constant_to_init
import convert_gather_to_init
import convert_unsuqeeze_to_init
import convert_concat_to_init
import convert_reduceMean_to_globalAveragePool
import fuse_pad_into_averagePool
import fuse_pad_into_maxPool
import fuse_pad_into_conv
