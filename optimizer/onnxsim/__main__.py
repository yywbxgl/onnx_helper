import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '...'))

import argparse
import onnx     # type: ignore
import optimizer.onnxsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                        nargs='?', type=int, default=0)
    parser.add_argument('--skip-optimization', help='Skip optimization of ONNX optimizers.',
                        action='store_true')
    parser.add_argument(
        '--input-shape', help='The manually-set static input shape, useful when the input shape is dynamic. The value should be "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.', type=str, nargs='+')
    args = parser.parse_args()
    print("Simplifying...")
    input_shapes = {}
    if args.input_shape is not None:
        for x in args.input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape
    model_opt = optimizer.onnxsim.simplify(
        args.input_model, check_n=args.check_n, perform_optimization=not args.skip_optimization, input_shapes=input_shapes)

    onnx.save(model_opt, args.output_model)
    print("Ok!")


if __name__ == '__main__':
    main()