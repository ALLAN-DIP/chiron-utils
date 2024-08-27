from chiron_utils.bots.baseline_models.constants import *
from chiron_utils.bots.baseline_models.preprocess import encode_class, decode_class, entry_to_vectors

class Results():
    def __init__(self, models, split_phase_types):
        self.models = models
        self.split_phase_types = split_phase_types

        self.class_corrects = dict()
        self.class_totals = dict()
        self.class_accuracies = dict()

        self.all_correct = 0
        self.all_total = 0
        self.all_accuracy = None

    def __repr__(self):
        output = f"Complete Correct: {self.all_correct}\nComplete Total: {self.all_total}\nComplete Accuracy: {(100 * self.all_accuracy):.2f}%"
        if self.split_phase_types:
            for phase_type in self.class_accuracies.keys():
                output += f"\nClass Correct ({phase_type}): {(self.class_corrects[phase_type])}\n"
                output += f"Class Total ({phase_type}): {(self.class_totals[phase_type])}\n"
                output += f"Class Accuracy ({phase_type}): {(100 * self.class_accuracies[phase_type]):.2f}%\n"
        return output

    def evaluate(self, test_dict):
        for phase_type, data in test_dict.items():
            print(f"Predicting for phases {phase_type}")
            pred_labels = self.models[phase_type].predict(data[0])
            pred_orders = map(decode_class, pred_labels)
            true_orders = map(decode_class, data[1])

            class_correct = 0
            class_total = 0

            for pred, true in zip(pred_orders, true_orders):
                correct, total = order_accuracy(pred, true)
                class_correct += correct
                class_total += total

            self.class_corrects[phase_type] = class_correct
            self.class_totals[phase_type] = class_total
            self.class_accuracies[phase_type] = class_correct / class_total
            self.all_correct += class_correct
            self.all_total += class_total

        self.all_accuracy = self.all_correct / self.all_total

def order_accuracy(predicted, true):
    correct = 0
    total = 0

    for i, power in enumerate(POWERS):
        for order in predicted[i]:
            correct += order in true[i]
            total += 1
        for order in true[i]:
            correct += order in predicted[i]
            total += 1
    return correct, total

def evaluate_model(test_dict, models, split_phase_types=False):
    results = Results(models, split_phase_types)
    results.evaluate(test_dict)
    return results

def infer(model, entry_vector):
    print(entry_vector)
    pred_labels = model.predict(entry_vector)
    print(pred_labels)
    pred_orders = decode_class(pred_labels[0])
    return pred_orders