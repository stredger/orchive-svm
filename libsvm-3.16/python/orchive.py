
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

    train = {'heads':[], 'attrs':[], 'class':[]}
    classify = {'heads':[], 'attrs':[], 'class':[]}

    if len(l) < n:
        print "list length larger than fold! exiting"
        sys.exit()

    for i in xrange(len(l)):
        if i % n:
            classify.append(l[i])
        else:
            train.append(l[i])

    return train, classify





h, a, c = read_csv('orchive_svm_25-03-2013.csv')


ht, hc = partition_list(h)
at, ac = partition_list(a)
ct, cc = partition_list(c)

m = svm_train(ct, at)
resutls = svm_predict([cc[0]],[ac[0]], m, '-b 0')

print results

