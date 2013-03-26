
import sys

from svmutil import *

CALL_ONE = 'N01'
CALL_TWO = 'N04'



def _to_floats(l):

    for i in xrange(len(l)):
        l[i] = float(l[i])

    return l


def read_csv(fname):

    headers = []
    attribs = []
    classes = []

    f = open(fname)
    
    for line in f.readlines():
        line = line.strip().split(',')
        headers.append(line[:3])
        classes.append(line[-1])
        attribs.append(line[3:-1])

    f.close()

    classes = _to_floats(classes)
    for i in xrange(len(attribs)):
        attribs[i] = _to_floats(attribs[i])

    return (headers, attribs, classes)



def partition_lists(h, a, c, n=10):

    id_col = 1

    train = {'heads':[], 'attrs':[], 'class':[]}
    classify = {'heads':[], 'attrs':[], 'class':[]}

    if len(h) < n:
        print "list length smaller than fold! exiting"
        sys.exit()

    for i in xrange(len(h)):
        if int(h[i][id_col]) % n:
            classify['heads'].append(h[i])
            classify['attrs'].append(a[i])
            classify['class'].append(c[i])
        else:
            train['heads'].append(h[i])
            train['attrs'].append(a[i])
            train['class'].append(c[i])

    return train, classify


h, a, c = read_csv('orchive_svm_25-03-2013.csv')


training, classify = partition_lists(h, a, c)
#print len(training['heads']), len(classify['heads'])


m = svm_train(training['class'], training['attrs'])
results = svm_predict(classify['class'],classify['attrs'], m)

print results

