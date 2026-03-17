import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def loadMetricsFile(filePath):
    # Load one metrics JSON file and return it as a dictionary
    with open(filePath, "r", encoding="utf-8") as metricsFile:
        metrics = json.load(metricsFile)

    return metrics


def buildMetricsSummary():
    # Load metrics from all three providers
    baselineMetrics = loadMetricsFile("outputs/metrics/baseline_metrics.json")
    nbMetrics = loadMetricsFile("outputs/metrics/nb_metrics.json")
    logisticMetrics = loadMetricsFile("outputs/metrics/logistic_metrics.json")

    # Put the key values into one table for easier charting
    summaryRows = [
        {
            "provider": baselineMetrics["provider"],
            "priority_accuracy": baselineMetrics["priority_accuracy"],
            "priority_macro_f1": baselineMetrics["priority_macro_f1"],
            "priority_weighted_f1": baselineMetrics["priority_weighted_f1"],
            "action_required_f1": baselineMetrics["action_required_f1"]
        },
        {
            "provider": nbMetrics["provider"],
            "priority_accuracy": nbMetrics["priority_accuracy"],
            "priority_macro_f1": nbMetrics["priority_macro_f1"],
            "priority_weighted_f1": nbMetrics["priority_weighted_f1"],
            "action_required_f1": nbMetrics["action_required_f1"]
        },
        {
            "provider": logisticMetrics["provider"],
            "priority_accuracy": logisticMetrics["priority_accuracy"],
            "priority_macro_f1": logisticMetrics["priority_macro_f1"],
            "priority_weighted_f1": logisticMetrics["priority_weighted_f1"],
            "action_required_f1": logisticMetrics["action_required_f1"]
        }
    ]

    summaryDataFrame = pd.DataFrame(summaryRows)
    return summaryDataFrame


def createMetricChart(summaryDataFrame, metricColumnName, chartTitle, outputFilePath):
    # Create a simple bar chart for one metric
    plt.figure(figsize=(8, 5))
    plt.bar(summaryDataFrame["provider"], summaryDataFrame[metricColumnName])
    plt.title(chartTitle)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outputFilePath)
    plt.close()


def createErrorChart(errorSummaryDataFrame, outputFilePath):
    # Create a grouped bar chart showing total, priority, and action errors
    providers = errorSummaryDataFrame["provider"]
    xPositions = range(len(providers))
    barWidth = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(
        [x - barWidth for x in xPositions],
        errorSummaryDataFrame["total_error_rows"],
        width=barWidth,
        label="Total error rows"
    )

    plt.bar(
        xPositions,
        errorSummaryDataFrame["priority_errors"],
        width=barWidth,
        label="Priority errors"
    )

    plt.bar(
        [x + barWidth for x in xPositions],
        errorSummaryDataFrame["action_errors"],
        width=barWidth,
        label="Action errors"
    )

    plt.title("Error Comparison by Provider")
    plt.ylabel("Count")
    plt.xticks(list(xPositions), providers, rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputFilePath)
    plt.close()


def createPerClassF1Chart(outputFilePath):
    # Load the classification reports from each metrics file
    baselineMetrics = loadMetricsFile("outputs/metrics/baseline_metrics.json")
    nbMetrics = loadMetricsFile("outputs/metrics/nb_metrics.json")
    logisticMetrics = loadMetricsFile("outputs/metrics/logistic_metrics.json")

    classNames = ["urgent", "important", "routine", "informational"]

    baselineScores = []
    nbScores = []
    logisticScores = []

    for className in classNames:
        baselineScores.append(
            baselineMetrics["priority_classification_report"].get(className, {}).get("f1-score", 0)
        )
        nbScores.append(
            nbMetrics["priority_classification_report"].get(className, {}).get("f1-score", 0)
        )
        logisticScores.append(
            logisticMetrics["priority_classification_report"].get(className, {}).get("f1-score", 0)
        )

    xPositions = range(len(classNames))
    barWidth = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(
        [x - barWidth for x in xPositions],
        baselineScores,
        width=barWidth,
        label="baseline_rules"
    )

    plt.bar(
        xPositions,
        nbScores,
        width=barWidth,
        label="naive_bayes"
    )

    plt.bar(
        [x + barWidth for x in xPositions],
        logisticScores,
        width=barWidth,
        label="logistic_regression"
    )

    plt.title("Per-Class Priority F1 Comparison")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(list(xPositions), classNames)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputFilePath)
    plt.close()


if __name__ == "__main__":
    os.makedirs("outputs/charts", exist_ok=True)

    # Build the summary table from metrics files
    summaryDataFrame = buildMetricsSummary()

    # Save the summary table itself for reference
    summaryDataFrame.to_csv("outputs/metrics/metrics_summary.csv", index=False)

    # Create metric comparison charts
    createMetricChart(
        summaryDataFrame,
        "priority_accuracy",
        "Priority Accuracy Comparison",
        "outputs/charts/priority_accuracy_comparison.png"
    )

    createMetricChart(
        summaryDataFrame,
        "priority_macro_f1",
        "Priority Macro F1 Comparison",
        "outputs/charts/priority_macro_f1_comparison.png"
    )

    createMetricChart(
        summaryDataFrame,
        "priority_weighted_f1",
        "Priority Weighted F1 Comparison",
        "outputs/charts/priority_weighted_f1_comparison.png"
    )

    createMetricChart(
        summaryDataFrame,
        "action_required_f1",
        "Action Required F1 Comparison",
        "outputs/charts/action_required_f1_comparison.png"
    )

    # Load error summary and create error chart
    errorSummaryDataFrame = pd.read_csv("outputs/analysis/error_summary.csv")

    createErrorChart(
        errorSummaryDataFrame,
        "outputs/charts/error_comparison.png"
    )

    # Create per-class F1 chart
    createPerClassF1Chart(
        "outputs/charts/per_class_priority_f1_comparison.png"
    )

    print("Visualisations created successfully.")
    print("Charts saved in: outputs/charts/")
    print("\nCreated files:")
    print("- outputs/charts/priority_accuracy_comparison.png")
    print("- outputs/charts/priority_macro_f1_comparison.png")
    print("- outputs/charts/priority_weighted_f1_comparison.png")
    print("- outputs/charts/action_required_f1_comparison.png")
    print("- outputs/charts/error_comparison.png")
    print("- outputs/charts/per_class_priority_f1_comparison.png")
    print("- outputs/metrics/metrics_summary.csv")