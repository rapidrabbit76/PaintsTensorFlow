from Training.PaintsTensorFlowDraftModelTraining import PaintsTensorFlowDraftModelTrain
from Training.PaintsTensorFlowTraining import PaintsTensorFlowTrain
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-loadEpochs", "--loadEpochs", default=0)
    parser.add_argument("-mode", "--mode")
    args = parser.parse_args()
    loadEpochs = args.loadEpochs
    mode = str(args.mode)

    if mode == "draft":
        model = PaintsTensorFlowDraftModelTrain()
        model.training(loadEpochs=loadEpochs)
    elif mode == "512":
        model = PaintsTensorFlowTrain()
        model.training(loadEpochs=loadEpochs)
    else:
        message = "Select [\"draft\",\"512\"] '{}' is not".format(mode)
        raise NotImplementedError(message)
