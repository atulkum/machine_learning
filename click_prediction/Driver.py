from LR import LR
import sys
from Util import Util
from PROBIT_R import PROBIT_R

def main():
     
    algo = sys.argv[1]
    trainfile = sys.argv[2]
    weightfile = sys.argv[3]
    testfilename = sys.argv[4]
    isCr = sys.argv[5]
    
    if(algo == 'LR'):
        D = 2**20   
        lr = LR(D)
        lr.train(trainfile)
        lr.saveWeight(weightfile)
        #lr.readWeight(weightfile)
        if(isCr == 'cr'):
            Util.cross_validation(testfilename, lr.get_features, lr.get_prediction)
    elif(algo == 'LR_CONV'):
        D = 2**30   
        lr = LR(D)
        lr.train_conjecture(trainfile)
        lr.saveWeight(weightfile)
        if(isCr == 'cr'):
            Util.cross_validation(testfilename, lr.get_features_conjectured, lr.get_prediction)
        else:
            Util.test(testfilename, 'LR_CONV_TEST', lr.get_features_conjectured, lr.get_prediction)
    elif(algo == 'PROBIT'):
        D = 2**20   
        pr = PROBIT_R(D, 0.3, 0.05, 0.01)
        pr.train(trainfile)
        pr.saveWeight(weightfile)
        if(isCr == 'cr'):
            Util.cross_validation(testfilename, pr.get_features, pr.predict)
        else:
            Util.test(testfilename, 'PROBIT_TEST',pr.get_features, pr.predict)

if __name__ == '__main__': 
    main()