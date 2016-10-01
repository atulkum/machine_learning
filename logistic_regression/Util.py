from datetime import datetime
from csv import DictReader
from math import exp, log

class Util(object):
    def test(testfile, base_filename, get_feature, get_prediction):
        with open(str(datetime.now()) + base_filename, 'w') as submission:
            submission.write('Id,Predicted\n')
            for t, row in enumerate(DictReader(open(testfile))):
                Id = row['Id']
                del row['Id']
                x = get_feature(row)
                prediction = get_prediction(x)
                submission.write('%s,%f\n' % (Id, prediction))


    def sigmoid(a):
        return 1. / (1. + exp(-max(min(a, 20.), -20.))) 
    
    def logloss(p, y):
        p = max(min(p, 1. - 10e-12), 10e-12)
        return -log(p) if y == 1. else -log(1. - p)
    
    def cross_validation(testfilename,get_feature, get_prediction): 
        lossTest = 0.
        count = 0
        for t, row in enumerate(DictReader(open(testfilename))):
            y = 1. if row['Label'] == '1' else 0.
            del row['Label']  # can't let the model peek the answe
            del row['Id']
            x = get_feature(row)
            prediction = get_prediction(x)
            lossTest += Util.logloss(prediction, y)
            count += 1
    
        print('logloss:%f' % (lossTest/count))
