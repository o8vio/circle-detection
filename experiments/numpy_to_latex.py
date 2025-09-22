import numpy as np

def scores_to_latex(scores, param_values, param_name, 
                    featurization_names=("Betti Curves", "Mix de features"),
                    metric_names=("Accuracy", "Brier Score Loss"),
                    score_names=("training score", "validation score"),
                    split_by="metric"):
    """
    Convert classification experiment results into two LaTeX tables.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (num_param_values, num_seeds, 2, 2, 2).
    param_values : list or array
        Values of the varied parameter, one per row.
    param_name : str
        Name of the parameter for the first column.
    featurization_names : tuple of str
        Names of the two vectorizations.
    metric_names : tuple of str
        Names of the two metrics (e.g., Accuracy, Brier).
    score_names : tuple of str
        Names of the two score types (e.g., Train, Test).
    split_by : str
        Which dimension to split tables on: "metric", "vectorization", or "score".
    """
    num_param_values = scores.shape[0]

    # compute mean ± std across seeds
    means = scores.mean(axis=1)
    stds = scores.std(axis=1)

    # Decide split axis
    if split_by == "featurization":
        split_axis = 1  # index over featurization_names
        split_labels = featurization_names
    elif split_by == "metric":
        split_axis = 2  # index over metric_names
        split_labels = metric_names
    elif split_by == "score_type":
        split_axis = 3  # index over score_names
        split_labels = score_names
    else:
        raise ValueError("split_by must be 'metric', 'featurization', or 'score_type'")

    tables = []

    for split_idx, split_label in enumerate(split_labels):
        
        header_cols = []
        
        if split_by == "metric":
            for feat in featurization_names:
                for sc in score_names:
                    header_cols.append(f"{vec} {sc}")
        elif split_by == "vectorization":
            for met in metric_names:
                for sc in score_names:
                    header_cols.append(f"{met} {sc}")
        elif split_by == "score":
            for vec in vectorization_names:
                for met in metric_names:
                    header_cols.append(f"{vec} {met}")

        latex = []
        latex.append("\\begin{tabular}{l" + "c"*len(header_cols) + "}")
        latex.append("\\hline")
        latex.append(param_name + " & " + " & ".join(header_cols) + r" \")
        latex.append("\\hline")

        for i, val in enumerate(param_values):
            row = [str(val)]
            for vi in range(2):
                for mi in range(2):
                    for si in range(2):
                        if split_by == "metric" and mi != split_idx:
                            continue
                        if split_by == "vectorization" and vi != split_idx:
                            continue
                        if split_by == "score" and si != split_idx:
                            continue
                        mean = means[i, vi, mi, si]
                        std = stds[i, vi, mi, si]
                        row.append(f"{mean:.3f} $\\pm$ {std:.3f}")
            latex.append(" & ".join(row) + r" \")

        latex.append("\\hline")
        latex.append("\\end{tabular}")

        tables.append((split_label, "\n".join(latex)))

    return tables


# Example usage
if __name__ == "__main__":
    num_param_values, num_seeds = 3, 90
    # fake data
    scores = np.random.rand(num_param_values, num_seeds, 2, 2, 2)
    param_values = [10, 50, 100]

    tables = scores_to_latex(scores, param_values, param_name="Samples", split_by="metric")
    for label, table in tables:
        print(f"% Table for {label}")
        print(table)
        print("\n")
