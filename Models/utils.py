import json, os
from venv import create
import pandas as pd
from joblib import dump, load
from embeddings_loader import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import sys

def get_f1_test_scores(filename, score_list, embeds):
    with open(filename, "r") as f:
        x = json.loads(f.read())
    result = []
    c = 0
    for each_cell in x["cells"]:
        if "outputs" in each_cell and each_cell["outputs"]:
            try:
                temp = [filename.replace(".ipynb", '') + "_" + embeds[c]]
                output = each_cell["outputs"][0]["text"]
                for each_output in output:
                    for score in score_list:
                        if score in each_output:
                            test_score = each_output.replace(f"{score}:  ", '')
                            test_score = test_score[:-2]
                            temp.append(float(test_score))
                if len(temp)>1:
                    result.append(temp)
                    c+=1
            except:
                continue
    return result

parent_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
save_folder_path = os.path.join(parent_dir, 'Models\ModelDumps')

def save_model(model, save_path):
    dump(model, os.path.join(save_folder_path, save_path))

def load_model(model_path):
    return load(os.path.join(save_folder_path, model_path))

try:
    print(sys.argv[1])
    train_labels, dev_labels = load_labels(sys.argv[1])
except:
    train_labels, dev_labels = load_labels()

label_replacement = {
    'OFF': 0,
    'NOT': 1,
}

# Replace labels with numbers
train_labels = [label_replacement[label] for label in train_labels]
dev_labels = [label_replacement[label] for label in dev_labels]

def computeAllScores(y_pred_train, y_pred_dev):
    print("Accuracy Train: ", accuracy_score(train_labels, y_pred_train))
    print("Accuracy Dev: ", accuracy_score(dev_labels, y_pred_dev))

    print("Weighted F1 Train: ", f1_score(train_labels, y_pred_train, average='weighted'))
    print("Weighted F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='weighted'))

    print("Macro F1 Train: ", f1_score(train_labels, y_pred_train, average='macro'))
    print("Macro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='macro'))

    print("Micro F1 Train: ", f1_score(train_labels, y_pred_train, average='micro'))
    print("Micro F1 Dev: ", f1_score(dev_labels, y_pred_dev, average='micro'))

    print("Weighted Recall Train: ", recall_score(train_labels, y_pred_train, average='weighted'))
    print("Weighted Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='weighted'))

    print("Macro Recall Train: ", recall_score(train_labels, y_pred_train, average='macro'))
    print("Macro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='macro'))

    print("Micro Recall Train: ", recall_score(train_labels, y_pred_train, average='micro'))
    print("Micro Recall Dev: ", recall_score(dev_labels, y_pred_dev, average='micro'))

    # Confusion Matrix
    print("Confusion Matrix Train: ")
    print(confusion_matrix(train_labels, y_pred_train))
    print("Confusion Matrix Dev: ")
    print(confusion_matrix(dev_labels, y_pred_dev))
