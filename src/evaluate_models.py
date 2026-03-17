import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def normaliseTextValue(value):
    # Convert values into clean lowercase text so comparisons are consistent
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def loadPredictionFile(filePath):
    # Load a prediction CSV file
    dataFrame = pd.read_csv(filePath)

    # Normalise the true and predicted columns to avoid type/capitalisation issues
    dataFrame["priority_label"] = dataFrame["priority_label"].apply(normaliseTextValue)
    dataFrame["predicted_priority_label"] = dataFrame["predicted_priority_label"].apply(normaliseTextValue)
    dataFrame["action_required"] = dataFrame["action_required"].apply(normaliseTextValue)
    dataFrame["predicted_action_required"] = dataFrame["predicted_action_required"].apply(normaliseTextValue)

    return dataFrame


def evaluatePredictionFile(providerName, filePath, outputMetricsPath):
    # Load the predictions
    dataFrame = loadPredictionFile(filePath)

    # Extract true and predicted values for the main priority task
    truePriorityLabels = dataFrame["priority_label"]
    predictedPriorityLabels = dataFrame["predicted_priority_label"]

    # Extract true and predicted values for the action_required task
    trueActionLabels = dataFrame["action_required"]
    predictedActionLabels = dataFrame["predicted_action_required"]

    # Calculate main priority metrics
    priorityAccuracy = accuracy_score(truePriorityLabels, predictedPriorityLabels)
    priorityMacroF1 = f1_score(truePriorityLabels, predictedPriorityLabels, average="macro")
    priorityWeightedF1 = f1_score(truePriorityLabels, predictedPriorityLabels, average="weighted")

    # Calculate action_required metric
    actionRequiredF1 = f1_score(trueActionLabels, predictedActionLabels, average="binary", pos_label="true")

    # Build a full per-class classification report for the priority task
    priorityClassReport = classification_report(
        truePriorityLabels,
        predictedPriorityLabels,
        output_dict=True,
        zero_division=0
    )

    # Store all metrics in one dictionary
    metrics = {
        "provider": providerName,
        "rows_evaluated": len(dataFrame),
        "priority_accuracy": priorityAccuracy,
        "priority_macro_f1": priorityMacroF1,
        "priority_weighted_f1": priorityWeightedF1,
        "action_required_f1": actionRequiredF1,
        "priority_classification_report": priorityClassReport
    }

    # Save metrics as JSON
    with open(outputMetricsPath, "w", encoding="utf-8") as metricsFile:
        json.dump(metrics, metricsFile, indent=4)

    return metrics


if __name__ == "__main__":
    baselineMetrics = evaluatePredictionFile(
        "baseline_rules",
        "outputs/predictions/baseline_predictions.csv",
        "outputs/metrics/baseline_metrics.json"
    )

    nbMetrics = evaluatePredictionFile(
        "naive_bayes",
        "outputs/predictions/nb_predictions.csv",
        "outputs/metrics/nb_metrics.json"
    )

    logisticMetrics = evaluatePredictionFile(
        "logistic_regression",
        "outputs/predictions/logistic_predictions.csv",
        "outputs/metrics/logistic_metrics.json"
    )

    print("Evaluation complete.")
    print("\nSummary:")

    for metrics in [baselineMetrics, nbMetrics, logisticMetrics]:
        print(f"\nProvider: {metrics['provider']}")
        print(f"Rows evaluated: {metrics['rows_evaluated']}")
        print(f"Priority accuracy: {metrics['priority_accuracy']:.4f}")
        print(f"Priority macro F1: {metrics['priority_macro_f1']:.4f}")
        print(f"Priority weighted F1: {metrics['priority_weighted_f1']:.4f}")
        print(f"Action required F1: {metrics['action_required_f1']:.4f}")