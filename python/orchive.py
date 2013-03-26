
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
        if not int(h[i][id_col]) % n:
            classify['heads'].append(h[i])
            classify['attrs'].append(a[i])
            classify['class'].append(c[i])
        else:
            train['heads'].append(h[i])
            train['attrs'].append(a[i])
            train['class'].append(c[i])

    return train, classify


# Takes in a list of class values and returns the majority value
def determine_majority(c):
    sum_neg = 0
    sum_pos = 0

    for i in c:
        if i == -1:
            sum_neg += 1
        else:
            sum_pos += 1

    if sum_neg > sum_pos:
        return -1
    else:
        return 1


h, a, c = read_csv('orchive_svm_25-03-2013.csv')


training, classify = partition_lists(h, a, c, 2)
#print len(training['heads']), len(classify['heads'])

print len(training['class'])
print len(classify['class'])

m = svm_train(training['class'], training['attrs'])
results = svm_predict(classify['class'],classify['attrs'], m)



correct = 0
total = 0

print len(classify['heads']), len(results[0])

for i in range(len(results[0])):
    j = i
    while int(classify['heads'][j][1]) == int(classify['heads'][i][1]):
        if j == len(results[0])-1:
            break
        j += 1
    clip = results[0][i:j]
    print classify['heads'][i:j]
    print clip
    i = j
    print "i=", i

    '''predicted = determine_majority(results[0][i:i+sublistsize])
    real = determine_majority(classify['class'][i:i+sublistsize])
    if predicted == real:
        correct += 1
    total += 1
    #print results[0][i:i+sublistsize], predicted, real
print "Accuracy = " + str(100*float(correct)/total) + "% (" + str(correct) + "/" + str(total) + ")"
  '''  


