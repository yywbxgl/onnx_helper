import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from creator import ir_creator

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], "config_path  output_model")
        sys.exit(-1)
    
    # create model
    ir_creator.createModel(sys.argv[1], sys.argv[2])
