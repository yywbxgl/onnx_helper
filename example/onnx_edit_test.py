import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from creator import ir_creator

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print ("Usage:", sys.argv[0], " OnnxModel")
        sys.exit(-1)

    # export model
    ir_creator.exportModel(sys.argv[1], "out")

    # Then, you can change the config file and weight file
    
    # create model
    # ir_creator.createModel("out/bvlc_alexnet_cfg.txt", "out/")
    ir_creator.createModel("out/CNTKGraph_cfg.txt", "out/")
    #ir_creator.createModel("out/CNTKGraph_cfg.txt")
