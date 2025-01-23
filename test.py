import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_pag_metrics(true_pag, predicted_pags, true_labels, predicted_labels_list):
    metrics_list = []

    def adjacency_matrix_to_edges(matrix, labels):
        """ Convert adjacency matrix to edge list with label ordering """
        n = len(labels)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if matrix[i, j] > 0:  # if there is a directed edge
                    edges.append((labels[i], labels[j]))
                if matrix[j, i] > 0:  # for undirected edges
                    edges.append((labels[j], labels[i]))
        return set(edges)

    def structural_hamming_distance(edges_true, edges_pred):
        """ Structural Hamming Distance """
        return len(edges_true.symmetric_difference(edges_pred)) / len(edges_true.union(edges_pred))

    def false_discovery_rate(edges_true, edges_pred):
        """ False Discovery Rate """
        fp = len(edges_pred - edges_true)
        tp = len(edges_true & edges_pred)
        return fp / (fp + tp) if (fp + tp) > 0 else 0

    def false_omission_rate(edges_true, edges_pred):
        """ False Omission Rate """
        fn = len(edges_true - edges_pred)
        tn = len(edges_true.union(edges_pred)) - len(edges_true & edges_pred)
        return fn / (fn + tn) if (fn + tn) > 0 else 0

    for pag, predicted_labels in zip(predicted_pags, predicted_labels_list):
        edges_true = adjacency_matrix_to_edges(true_pag, true_labels)
        edges_pred = adjacency_matrix_to_edges(pag, predicted_labels)

        shd = structural_hamming_distance(edges_true, edges_pred)
        fdr = false_discovery_rate(edges_true, edges_pred)
        for_ = false_omission_rate(edges_true, edges_pred)

        # Calculating precision, recall, and F1-score
        true_positive = len(edges_true & edges_pred)
        false_positive = len(edges_pred - edges_true)
        false_negative = len(edges_true - edges_pred)
        true_negative = len(edges_true.union(edges_pred)) - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "SHD": shd,
            "FDR": fdr,
            "FOR": for_,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }

        metrics_list.append(metrics)

    return metrics_list


true_pag = np.array([[0, 1, 0, 2, 0],
                     [1, 0, 2, 0, 0],
                     [0, 1, 0, 2, 2],
                     [1, 0, 2, 0, 2],
                     [0, 0, 2, 3, 0]])

predicted_pags = [
    np.array([[0, 0, 2, 2], [0, 0, 2, 0], [2, 1, 0, 2], [2, 0, 3, 0]]),
    np.array([[0, 2, 0, 0], [1, 0, 1, 0], [0, 2, 0, 1], [0, 0, 1, 0]])
]

true_labels = ['A', 'B', 'C', 'D', 'E']
predicted_labels_list = [['A', 'C', 'D', 'E'], ['C', 'D', 'E', 'B']]

metrics_list = calculate_pag_metrics(true_pag, predicted_pags, true_labels, predicted_labels_list)

# Create DataFrame from metrics for easy analysis
df = pd.DataFrame(metrics_list)

print(df)
