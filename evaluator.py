import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from dataloader import MRDataset


def extract_predictions(task, plane, train_initial, prefix_name = '', train=True):
    assert task in ['abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    models = os.listdir('./models/')

    model_name = list(filter(lambda name: prefix_name in name and task in name and plane in name, models))[-1]
    model_path = f'./models/{model_name}'
    print(model_path)

    mrnet = torch.load(model_path)
    _ = mrnet.eval()

    train_dataset = MRDataset('./data/', 
                              task, 
                              plane, 
                              train_initial,
                              train=train)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                            #    num_workers=11, 
                                               drop_last=False)

    predictions = []
    labels = []
    with torch.no_grad():
        for image, label, _ in tqdm(train_loader):
            # if torch.cuda.is_available():
            logit = mrnet(image)
            prediction = torch.sigmoid(logit)
            prediction_lst = list(prediction.tolist())[0]
            # print(label)
            # print(prediction_lst = prediction[0].item())
            # print(label[0][0][1].item())
            predictions.append(prediction_lst.index(max(prediction_lst)))
            labels.append(int(label[0][0][1].item()))

    return predictions, labels

def run(args):
    task = args.task
    prefix_name = '' if args.prefix_name is None else args.prefix_name
    train_initial = args.train_initial
    print("ARGURMENT: ", args.train_initial) ####bug here
    # Training set
    results = {}
    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, train_initial=train_initial, prefix_name=prefix_name)
        results['labels'] = labels
        results[plane] = predictions
        
    X = np.zeros((len(predictions), 3))
    X[:, 0] = results['axial']
    X[:, 1] = results['coronal']
    X[:, 2] = results['sagittal']
    # print(X)

    y = np.array(labels)
    # print(len(y))
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X, y)


    # Validation set
    results_val = {}
    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, train_initial=train_initial, prefix_name=prefix_name, train=False)
        results_val['labels'] = labels
        results_val[plane] = predictions

    X_val = np.zeros((len(predictions), 3))
    X_val[:, 0] = results_val['axial']
    X_val[:, 1] = results_val['coronal']
    X_val[:, 2] = results_val['sagittal']
    y_val = np.array(labels)

    y_pred_prob = logreg.predict_proba(X_val)[:, 1]
    y_pred = [1 if x >= 0.5 else 0 for x in y_pred_prob]

    # Metrics to print
    print(metrics.roc_auc_score(y_val, y_pred))
    print(metrics.confusion_matrix(y_val, y_pred))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('--prefix_name', type=str, required=False)
    parser.add_argument('--train_initial', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run(args)