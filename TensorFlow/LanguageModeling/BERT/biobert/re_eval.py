import os
import numpy as np
import pandas as pd
import sklearn.metrics
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str,  help='')
parser.add_argument('--answer_path', type=str,  help='')
parser.add_argument('--task', type=str,  default="binary", help='default:binary, possible other options:{chemprot}')
args = parser.parse_args()


testdf = pd.read_csv(args.answer_path, sep="\t", header=None)
preddf = pd.read_csv(args.output_path, sep="\t", header=None)


# binary
if args.task == "binary":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    pred_prob_one = [v[1] for v in pred]

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])
    results = dict()
    results["f1 score"] = f[1]
    results["recall"] = r[1]
    results["precision"] = p[1]
    results["specificity"] = r[0]

# chemprot
# micro-average of 5 target classes
# see "Potent pairing: ensemble of long short-term memory networks and support vector machine for chemical-protein relation extraction (Mehryary, 2018)" for details
if args.task == "chemprot":
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    str_to_int_mapper = dict()

    testdf.iloc[:,3] = testdf.iloc[:, 3].fillna("False")
    for i,v in enumerate(sorted(testdf.iloc[:,3].unique())):
        str_to_int_mapper[v] = i
    test_answer = [str_to_int_mapper[v] for v in testdf.iloc[:,3]]

    p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_class, y_true=test_answer, labels=[0,1,2,3,4], average="micro")
    results = dict()
    results["f1 score"] = f
    results["recall"] = r
    results["precision"] = p

for k,v in results.items():
    print("{:11s} : {:.2%}".format(k,v))
