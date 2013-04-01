
import sys, random, textwrap

from svmutil import *
from optparse import OptionParser

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


# Partitions the data using the mod operator
def partition_lists(h, a, c, ratio):

    id_col = 1

    train = {'heads':[], 'attrs':[], 'class':[]}
    classify = {'heads':[], 'attrs':[], 'class':[]}

    # When the ratio (i.e. percentage) is above 0.5, everything basically rounds
    # to 1, which produces bad results when using mod. Therefore, when ratio > 0.5,
    # use (1-ratio) and do the opposite i.e. "not"
    if ratio > 0.5:
        n = int(1.0/(1.0-ratio))

        for i in xrange(len(h)):
            if not int(h[i][id_col]) % n:
                classify['heads'].append(h[i])
                classify['attrs'].append(a[i])
                classify['class'].append(c[i])
            else:
                train['heads'].append(h[i])
                train['attrs'].append(a[i])
                train['class'].append(c[i])
    else:
        n = int(1.0/ratio)

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


# Randomly partitions the data by shuffling a list of call IDs, then splitting between
# training and classification depending on the ratio
def random_partition_lists(h, a, c, ratio):  

    id_col = 1
    
    train_contains_neg = False
    train_contains_pos = False

    # Loop until the random partitioning produces a training set that inclues both -1 and +1 classes
    while train_contains_neg == False or train_contains_pos == False:

        train_contains_neg = False
        train_contains_pos = False

        training = {'heads':[], 'attrs':[], 'class':[]}
        classify = {'heads':[], 'attrs':[], 'class':[]}

        training_ids = []

        for i in range(len(h)):
            if h[i][id_col] not in training_ids:
                training_ids.append(h[i][id_col])
        random.shuffle(training_ids)

        cutoff = int(ratio*len(training_ids))
        training_ids = training_ids[0:cutoff]

        for i in range(len(h)):
            if h[i][id_col] in training_ids:
                training['heads'].append(h[i])
                training['attrs'].append(a[i])
                training['class'].append(c[i])
            else:  
                classify['heads'].append(h[i])
                classify['attrs'].append(a[i])
                classify['class'].append(c[i])


        for i in range(0, len(training['class'])):
            if training['class'][i] == -1:
                train_contains_neg = True
            if training['class'][i] == 1:
                train_contains_pos = True

        # If the ratio causes there to be less than two training calls, do not
        # require that the training set contains both a -1 and +1 class
        if len(training_ids) < 2:
            train_contains_neg = True
            train_contains_pos = True

    return training, classify


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


# Prints out info about the paritioning of the data
def print_partition_info(training, classify):
    id_col = 1

    training_ids = []
    training_neg_clip_count = 0
    training_pos_clip_count = 0
    for i in range(0, len(training['heads'])):
        if training['heads'][i][id_col] not in training_ids:
            training_ids.append(training['heads'][i][id_col])
        if training['class'][i] == -1:
            training_neg_clip_count += 1
        if training['class'][i] == 1:
            training_pos_clip_count += 1
            

    classify_ids = []
    classify_neg_clip_count = 0
    classify_pos_clip_count = 0
    for i in range(0, len(classify['heads'])):
        if classify['heads'][i][id_col] not in classify_ids:
            classify_ids.append(classify['heads'][i][id_col])
        if classify['class'][i] == -1:
            classify_neg_clip_count  += 1
        if classify['class'][i] == 1:
            classify_pos_clip_count += 1

    print "Training Examples:"
    print "------------------"
    print "No. of calls \t", len(training_ids)
    print "No. of clips \t", len(training['class'])
    print "Call IDs: ", textwrap.fill(str(training_ids), initial_indent='\t', subsequent_indent='\t\t')
    print "No. of -1s\t", training_neg_clip_count, "clips"
    print "No. of +1s\t", training_pos_clip_count, "clips"
    print

    print "Classifying Examples:"
    print "---------------------"
    print "No. of calls \t", len(classify_ids)
    print "No. of clips \t", len(classify['class'])
    print "Call IDs: ", textwrap.fill(str(classify_ids), initial_indent='\t', subsequent_indent='\t\t')
    print "No. of -1s\t", classify_neg_clip_count, "clips"
    print "No. of +1s\t", classify_pos_clip_count, "clips"    
    print   


# This is called for each run of the program
def run_program(input_file, options):
    h, a, c = read_csv(input_file)

    if options.partitioning.lower() == "random":
        training, classify = random_partition_lists(h, a, c, options.ratio)
    else:
        training, classify = partition_lists(h, a, c, options.ratio)

    if options.verbose >= 2:
        print_partition_info(training, classify)

    if options.verbose == 3:
        m = svm_train(training['class'], training['attrs'])
        results = svm_predict(classify['class'],classify['attrs'], m)
    else:
        m = svm_train(training['class'], training['attrs'], '-q')
        results = svm_predict(classify['class'],classify['attrs'], m, '-q')

    call_correct = 0
    call_total = 0

    i = 0
    while i < len(results[0]):
        j = i
        while j < len(results[0]) and int(classify['heads'][j][1]) == int(classify['heads'][i][1]):
            j += 1
        predicted = results[0][i:j]
        actual = classify['class'][i:j]
        i = j

        predicted_final = determine_majority(predicted)
        actual_final = determine_majority(actual)
        if predicted_final == actual_final:
            call_correct += 1
        call_total += 1

    return call_correct, call_total


# ===================
# Program entry point
# ===================

# Set up command line options
usage = "usage: %prog [options] filename \n\nfilename: name of the input file"
parser = OptionParser(usage=usage)
parser.add_option("-r", "--ratio", dest="ratio", metavar="RATIO", type="float", help="percentage of examples to use for training: <0.0-1.0>, i.e. 0.9 ==> 90% training, 10% classification [default: %default]", default=0.9)
parser.add_option("-p", "--partition", dest="partitioning", metavar="METHOD", help="partition method: <mod> or <random> [default: %default]", default="mod")
parser.add_option("-n", "--runs", dest="runs", metavar="RUNS", type="int", help="number of times to run the program [default: %default]", default=1)
parser.add_option("-v", "--verbose", dest="verbose", metavar="LEVEL", type="int", help="verbose level: <0-3> [default: %default]", default=0)
(options, args) = parser.parse_args()

# Use this as default input file unless other given via command line
input_file = "orchive_svm_25-03-2013.csv"
if len(args) > 0:
    input_file = args[0]

call_accuracy_sum = 0.0

# Run program 'runs' number of times
for r in range(0, int(options.runs)):
    if options.verbose >= 2:
        print "==========\n" + "Run #" + str(r + 1) + "\n=========="
    call_correct, call_total = run_program(input_file, options)
    call_accuracy = 100*float(call_correct)/float(call_total)
    call_accuracy_sum += call_accuracy
    if options.verbose >= 1:
        print "Call Accuracy = " + str(round(call_accuracy, 3)) + "% (" + str(call_correct) + "/" + str(call_total) + ")"
    if options.verbose >= 2:
        print

# Calculates and prints the average accuracy over all of the runs
if options.verbose >= 1:
    print "=======\nOVERALL\n======="
print "Average Accuracy = " + str(round(call_accuracy_sum/options.runs, 3)) + "% over " + str(options.runs) + " runs with a training:classification ratio of " + str(round(100*options.ratio, 1)) + "%:" + str(round(100*(1-options.ratio), 1)) + "%"
print

