import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from creator import ir_creator

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print ("Usage:", sys.argv[0], " OnnxModel   out_path")
        sys.exit(-1)

    # export model
    ir_creator.exportModel(sys.argv[1], sys.argv[2])

    # you can change the config file and weight file, Then create a new model
    

