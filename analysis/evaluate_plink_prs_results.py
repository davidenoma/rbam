import pandas as pd
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss
import matplotlib.pyplot as plt


def nagelkerke_r2(y_true, y_pred):
    """
    Computes Nagelkerke's pseudo-R² for binary traits.

    Parameters:
        y_true (array): True phenotype labels (0 = Control, 1 = Case)
        y_pred (array): Predicted PRS scores

    Returns:
        float: Nagelkerke's pseudo-R²
    """
    null_log_likelihood = log_loss(y_true, np.full_like(y_true, np.mean(y_true)))
    model_log_likelihood = log_loss(y_true, y_pred)
    r2 = 1 - (model_log_likelihood / null_log_likelihood)
    return max(0, r2)  # Ensure it's not negative

def compute_metrics(prs_file):
    """
    Computes AUC, Accuracy (threshold = 0.5), and Nagelkerke's R² from a PLINK PRS profile file.

    Parameters:
        prs_file (str): Path to the PRS profile file.
    """
    try:
        # Load PRS results
        prs_df = pd.read_csv(prs_file, delim_whitespace=True)
        print(f"Loaded PRS file: {prs_file}, {prs_df.shape[0]} individuals.")

        # Convert PHENO values: Case (2) → 1, Control (1) → 0
        prs_df["PHENO_BINARY"] = prs_df["PHENO"].replace({1: 0, 2: 1})

        # Extract true labels (y) and predicted scores
        y_true = prs_df["PHENO_BINARY"]
        y_scores = prs_df["SCORE"]

        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
        print(f"AUC: {auc:.4f}")

        # Compute Accuracy using threshold = 0.5
        y_pred = (y_scores > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy (Threshold = 0.5): {accuracy:.4f}")

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for PRS Model")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py <prs_file>")
        sys.exit(1)

    prs_file = sys.argv[1]
    compute_metrics(prs_file)
