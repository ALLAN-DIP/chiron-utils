from sklearn.neighbors import KNeighborsClassifier
import os
import json
import numpy as np
from constants import *
from preprocess import generate_x_y
from time import time
from evaluation import evaluate_model

def run_knn(train_path, test_path):
    train_dict = dict()
    k = 10
    split_phases = True

    print("Preprocessing training data")
    with open(train_path, 'r') as train:
        generate_x_y(train_dict, train, split_phase_types=split_phases)

    models = dict()
    print("Training models")
    for phase_type, data in train_dict.items():
        models[phase_type] = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='ball_tree', metric="hamming")
        models[phase_type].fit(data[0], data[1])
    
    print("Preprocessing testing data")
    test_dict = dict()
    with open(test_path, 'r') as test:
        generate_x_y(test_dict, test, split_phase_types=split_phases)
    
    print("Evaluating model")
    results = evaluate_model(test_dict, models, split_phase_types=split_phases)
    print(results)

def main():
    data_path = os.path.join("D:", os.sep, "Downloads", "dipnet-data-diplomacy-v1-27k-msgs", "test")
    train_path = os.path.join(data_path, "train.jsonl")
    test_path = os.path.join(data_path, "test.jsonl")

    run_knn(train_path, test_path)


if __name__ == "__main__":
    start_time = time()
    main()
    print(f"Total runtime: {(time() - start_time):.2f} seconds")