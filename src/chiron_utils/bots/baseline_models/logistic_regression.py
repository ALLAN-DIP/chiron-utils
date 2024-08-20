from sklearn.linear_model import LogisticRegression
from time import time
import os
import json
from preprocess import generate_x_y
from constants import *
from evaluation import evaluate_model

def order_accuracy(predicted, true):
    # print(f"Predicted: {predicted}\nTrue: {true}\n")
    macro_correct = 0
    macro_total = len(POWERS)
    micro_correct = 0
    micro_total = 0

    for i, power in enumerate(POWERS):
        macro_correct += predicted[i] == true[i]
        for order in predicted[i]:
            micro_correct += order in true[i]
            micro_total += 1
        for order in true[i]:
            micro_correct += order in predicted[i]
            micro_total += 1
    return macro_correct, micro_correct, macro_total, micro_total

def run_lr(train_path, test_path):
    train_dict = dict()
    split_phases = False

    print("Preprocessing training data")
    with open(train_path, 'r') as train:
        generate_x_y(train_dict, train, split_phase_types=split_phases)

    models = dict()
    print("Training models")
    for phase_type, data in train_dict.items():
        models[phase_type] = LogisticRegression(random_state=1)
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

    run_lr(train_path, test_path)

if __name__ == "__main__":
    start_time = time()
    main()
    print(f"Total runtime: {(time() - start_time):.2f} seconds")