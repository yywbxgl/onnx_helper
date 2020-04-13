from collections import OrderedDict

from typing import List, Dict, Union, Optional, Tuple
import copy

import onnx  # type: ignore
import onnx.helper  # type: ignore
import onnx.optimizer  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper
import onnxruntime as rt  # type: ignore

import numpy as np  # type: ignore

TensorShape = List[int]
TensorShapes = Dict[Optional[str], TensorShape]


def add_features_to_output(m: onnx.ModelProto, nodes: List[onnx.NodeProto]) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    :param nodes: nodes whose outputs will be added into the graph outputs
    """
    for node in nodes:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


def get_shape_from_value_info_proto(v: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in v.type.tensor_type.shape.dim]


def get_value_info_all(m: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    for v in m.graph.value_info:
        if v.name == name:
            return v

    for v in m.graph.input:
        if v.name == name:
            return v

    for v in m.graph.output:
        if v.name == name:
            return v

    return None


def get_shape(m: onnx.ModelProto, name: str) -> TensorShape:
    """
    Note: This method relies on onnx shape inference, which is not reliable. So only use it on input or output tensors
    """
    v = get_value_info_all(m, name)
    if v is not None:
        return get_shape_from_value_info_proto(v)
    raise RuntimeError('Cannot get shape of "{}"'.format(name))


def get_elem_type(m: onnx.ModelProto, name: str) -> Optional[int]:
    v = get_value_info_all(m, name)
    if v is not None:
        return v.type.tensor_type.elem_type
    return None


def get_np_type_from_elem_type(elem_type: int) -> int:
    sizes = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, str, np.bool,
             np.float16, np.double, np.uint32, np.uint64, np.complex64, np.complex128, np.float16)
    assert len(sizes) == 17
    size = sizes[elem_type]
    assert size is not None
    return size


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = list(set([ipt.name for ipt in model.graph.input]) -
                       set([x.name for x in model.graph.initializer]))
    return input_names


def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend(
                [onnx.ValueInfoProto(name=x.name,
                                     type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                           shape=shape)))])
    return model


def generate_rand_input(model, input_shapes: Optional[TensorShapes] = None):
    if input_shapes is None:
        input_shapes = {}
    input_names = get_input_names(model)
    full_input_shapes = {ipt: get_shape(model, ipt) for ipt in input_names}
    assert None not in input_shapes
    full_input_shapes.update(input_shapes)  # type: ignore
    for key in full_input_shapes:
        if np.prod(full_input_shapes[key]) <= 0:
            raise RuntimeError(
                'The shape of input "{}" has dynamic size, '
                'please determine the input size manually by --input-shape xxx'.format(key))

    inputs = {ipt: np.array(np.random.rand(*full_input_shapes[ipt]),
                            dtype=get_np_type_from_elem_type(get_elem_type(model, ipt))) for ipt in
              input_names}
    return inputs


def get_constant_nodes(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    const_tensors.extend([node.output[0]
                          for node in m.graph.node if node.op_type == 'Constant'])
    # If one of the input of a node is produced (directly or indirectly) by nms,
    # we consider the output of this node doesn't have constant shape,
    # so we do not simplify a such node even if the node is Shape op
    tensors_nms = []
    for node in m.graph.node:
        if any(x in tensors_nms for x in node.input):
            tensors_nms.extend(node.output)
        elif node.op_type == 'Shape':
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif node.op_type == 'NonMaxSuppression':
            tensors_nms.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return copy.deepcopy(const_nodes)


def forward(model, inputs=None, input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
    if input_shapes is None:
        input_shapes = {}
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    sess = rt.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=['CPUExecutionProvider'])
    if inputs is None:
        inputs = generate_rand_input(model, input_shapes=input_shapes)
    outputs = [x.name for x in sess.get_outputs()]
    run_options = rt.RunOptions()
    run_options.log_severity_level = 3
    res = OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))
    return res


def forward_for_node_outputs(model: onnx.ModelProto, nodes: List[onnx.NodeProto],
                             input_shapes: Optional[TensorShapes] = None) -> Dict[str, np.ndarray]:
    if input_shapes is None:
        input_shapes = {}
    model = copy.deepcopy(model)
    add_features_to_output(model, nodes)
    res = forward(model, input_shapes=input_shapes)
    return res


def insert_elem(repeated_container, index: int, element):
    repeated_container.extend([repeated_container[-1]])
    for i in reversed(range(index + 1, len(repeated_container) - 1)):
        repeated_container[i].CopyFrom(repeated_container[i - 1])
    repeated_container[index].CopyFrom(element)


def eliminate_const_nodes(model: onnx.ModelProto, const_nodes: List[onnx.NodeProto],
                          res: Dict[str, np.ndarray]) -> onnx.ModelProto:
    """
    :param model: the original onnx model
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: the simplified onnx model. Redundant ops are all removed.
    """
    for i, node in enumerate(model.graph.node):
        if node in const_nodes:
            for output in node.output:
                new_node = copy.deepcopy(node)
                new_node.name = "node_" + output
                new_node.op_type = 'Constant'
                new_attr = onnx.helper.make_attribute(
                    'value',
                    onnx.numpy_helper.from_array(res[output], name=output)
                    )
                del new_node.input[:]
                del new_node.attribute[:]
                del new_node.output[:]
                new_node.output.extend([output])
                new_node.attribute.extend([new_attr])
                insert_elem(model.graph.node, i + 1, new_node)
            del model.graph.node[i]

    return model


def optimize(model: onnx.ModelProto, skip_fuse_bn: bool) -> onnx.ModelProto:
    """
    :param model: The onnx model.
    :return: The optimized onnx model.
    Before simplifying, use this method to generate value_info, which is used in `forward_all`
    After simplifying, use this method to fold constants generated in previous step into initializer,
    and eliminate unused constants.
    """

    # Due to a onnx bug, https://github.com/onnx/onnx/issues/2417, we need to add missing initializers into inputs
    onnx.checker.check_model(model)
    input_num = len(model.graph.input)
    model = add_initializers_into_inputs(model)
    onnx.helper.strip_doc_string(model)
    onnx.checker.check_model(model)
    optimizers_list = ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                            # 'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                            # 'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                            # 'eliminate_nop_transpose', 'fuse_add_bias_into_conv', 
                                            # https://github.com/daquexian/onnx-simplifier/issues/31
                                            # 'fuse_consecutive_concats',
                                            # 'fuse_consecutive_log_softmax',
                                            # 'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes',
                                            # 'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm',
                                            # 'fuse_pad_into_conv', 
                                            # 'fuse_transpose_into_gemm'
                                            ]
    if not skip_fuse_bn:
        optimizers_list.append('fuse_bn_into_conv')

    model = onnx.optimizer.optimize(model, optimizers_list,
                                    fixed_point=True)
    del model.graph.input[input_num:]
    onnx.checker.check_model(model)
    return model


def check(model_opt: onnx.ModelProto, model_ori: onnx.ModelProto, n_times: int = 5,
          input_shapes: Optional[TensorShapes] = None) -> bool:
    """
    Warning: Some models (e.g., MobileNet) may fail this check by a small magnitude.
    Just ignore if it happens.
    :param input_shapes: Shapes of generated random inputs
    :param model_opt: The simplified ONNX model
    :param model_ori: The original ONNX model
    :param n_times: Generate n random inputs
    """
    if input_shapes is None:
        input_shapes = {}
    onnx.checker.check_model(model_opt)
    for i in range(n_times):
        print("Checking {}/{}...".format(i, n_times))
        rand_input = generate_rand_input(model_opt, input_shapes=input_shapes)
        res_opt = forward(model_opt, inputs=rand_input)
        res_ori = forward(model_ori, inputs=rand_input)

        for name in res_opt.keys():
            if not np.allclose(res_opt[name], res_ori[name], rtol=1e-4, atol=1e-5):
                print("Tensor {} changes after simplifying. The max diff is {}.".format(
                    name, np.max(np.abs(res_opt[name] - res_ori[name]))))
                print("Note that the checking is not always correct.")
                print("After simplifying:")
                print(res_opt[name])
                print("Before simplifying:")
                print(res_ori[name])
                print("----------------")
                return False
    return True


def clean_constant_nodes(const_nodes: List[onnx.NodeProto], res: Dict[str, np.ndarray]):
    """
    It seems not needed since commit 6f2a72, but maybe it still prevents some unknown bug
    :param const_nodes: const nodes detected by `get_constant_nodes`
    :param res: The dict containing all tensors, got by `forward_all`
    :return: The constant nodes which have an output in res
    """
    return [node for node in const_nodes if node.output[0] in res]


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: TensorShapes) -> TensorShapes:
    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(x))
    return input_shapes


def simplify(model: Union[str, onnx.ModelProto], check_n: int = 0, perform_optimization: bool = True,
        skip_fuse_bn: bool = False, input_shapes: Optional[TensorShapes] = None) \
        -> Tuple[onnx.ModelProto, bool]:
    if input_shapes is None:
        input_shapes = {}
    if type(model) == str:
        model = onnx.load(model)
    onnx.checker.check_model(model)
    model_ori = copy.deepcopy(model)
    model = onnx.shape_inference.infer_shapes(model)

    input_shapes = check_and_update_input_shapes(model, input_shapes)

    if perform_optimization:
        model = optimize(model, skip_fuse_bn)

    const_nodes = get_constant_nodes(model)
    res = forward_for_node_outputs(model, const_nodes, input_shapes=input_shapes)
    const_nodes = clean_constant_nodes(const_nodes, res)
    model = eliminate_const_nodes(model, const_nodes, res)
    onnx.checker.check_model(model)

    if perform_optimization:
        model = optimize(model, skip_fuse_bn)

    check_ok = check(model_ori, model, check_n, input_shapes=input_shapes)

    return model, check_ok
