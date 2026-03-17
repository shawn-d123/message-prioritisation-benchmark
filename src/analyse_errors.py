import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def normaliseTextValue(value):
    # Convert values into clean lowercase text for consistent comparisons
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def loadPredictionFile(filePath):
    # Load a prediction file and normalise key columns
    dataFrame = pd.read_csv(filePath)

    dataFrame["priority_label"] = dataFrame["priority_label"].apply(normaliseTextValue)
    dataFrame["predicted_priority_label"] = dataFrame["predicted_priority_label"].apply(normaliseTextValue)
    dataFrame["action_required"] = dataFrame["action_required"].apply(normaliseTextValue)
    dataFrame["predicted_action_required"] = dataFrame["predicted_action_required"].apply(normaliseTextValue)

    return dataFrame


def analysePredictionFile(providerName, filePath, outputFolderPath):
    # Load the predictions for one provider
    dataFrame = loadPredictionFile(filePath)

    # Create columns that show whether each prediction was correct
    dataFrame["priority_correct"] = dataFrame["priority_label"] == dataFrame["predicted_priority_label"]
    dataFrame["action_correct"] = dataFrame["action_required"] == dataFrame["predicted_action_required"]

    # Keep only rows where something went wrong
    errorDataFrame = dataFrame[
        (dataFrame["priority_correct"] == False) | (dataFrame["action_correct"] == False)
    ].copy()

    # Create a readable error type column
    errorTypes = []

    for _, row in errorDataFrame.iterrows():
        if row["priority_correct"] == False and row["action_correct"] == False:
            errorTypes.append("priority_and_action")
        elif row["priority_correct"] == False:
            errorTypes.append("priority_only")
        else:
            errorTypes.append("action_only")

    errorDataFrame["error_type"] = errorTypes

    # Save detailed error rows
    errorOutputFilePath = os.path.join(outputFolderPath, f"{providerName}_errors.csv")
    errorDataFrame.to_csv(errorOutputFilePath, index=False)

    # Build confusion pair counts for priority mistakes
    priorityErrorDataFrame = dataFrame[dataFrame["priority_correct"] == False].copy()

    if len(priorityErrorDataFrame) > 0:
        priorityErrorDataFrame["confusion_pair"] = (
            priorityErrorDataFrame["priority_label"] + "_predicted_as_" +
            priorityErrorDataFrame["predicted_priority_label"]
        )

        confusionPairCounts = (
            priorityErrorDataFrame["confusion_pair"]
            .value_counts()
            .reset_index()
        )
        confusionPairCounts.columns = ["confusion_pair", "count"]
    else:
        confusionPairCounts = pd.DataFrame(columns=["confusion_pair", "count"])

    confusionPairFilePath = os.path.join(outputFolderPath, f"{providerName}_confusion_pairs.csv")
    confusionPairCounts.to_csv(confusionPairFilePath, index=False)

    # Build a confusion matrix table for the priority task
    priorityLabels = ["urgent", "important", "routine", "informational"]

    confusionMatrixValues = confusion_matrix(
        dataFrame["priority_label"],
        dataFrame["predicted_priority_label"],
        labels=priorityLabels
    )

    confusionMatrixDataFrame = pd.DataFrame(
        confusionMatrixValues,
        index=priorityLabels,
        columns=priorityLabels
    )

    confusionMatrixFilePath = os.path.join(outputFolderPath, f"{providerName}_priority_confusion_matrix.csv")
    confusionMatrixDataFrame.to_csv(confusionMatrixFilePath)

    # Build a short summary for this provider
    summary = {
        "provider": providerName,
        "rows_evaluated": len(dataFrame),
        "total_error_rows": len(errorDataFrame),
        "priority_errors": int((dataFrame["priority_correct"] == False).sum()),
        "action_errors": int((dataFrame["action_correct"] == False).sum()),
        "priority_and_action_errors": int((errorDataFrame["error_type"] == "priority_and_action").sum()),
        "priority_only_errors": int((errorDataFrame["error_type"] == "priority_only").sum()),
        "action_only_errors": int((errorDataFrame["error_type"] == "action_only").sum())
    }

    return summary


if __name__ == "__main__":
    outputFolderPath = "outputs/analysis"
    os.makedirs(outputFolderPath, exist_ok=True)

    summaries = []

    summaries.append(
        analysePredictionFile(
            "baseline_rules",
            "outputs/predictions/baseline_predictions.csv",
            outputFolderPath
        )
    )

    summaries.append(
        analysePredictionFile(
            "naive_bayes",
            "outputs/predictions/nb_predictions.csv",
            outputFolderPath
        )
    )

    summaries.append(
        analysePredictionFile(
            "logistic_regression",
            "outputs/predictions/logistic_predictions.csv",
            outputFolderPath
        )
    )

    summaryDataFrame = pd.DataFrame(summaries)
    summaryFilePath = os.path.join(outputFolderPath, "error_summary.csv")
    summaryDataFrame.to_csv(summaryFilePath, index=False)

    print("Error analysis complete.")
    print(f"Summary saved to: {summaryFilePath}")
    print("\nError summary:")
    print(summaryDataFrame)