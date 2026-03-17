import pandas as pd


def normaliseTextValue(value):
    # Convert the input into clean lowercase text for easier keyword matching
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def predictActionRequired(messageText):
    # Convert the message into normalised lowercase text
    cleanedMessageText = normaliseTextValue(messageText)

    # Keywords that often suggest the recipient needs to do something
    actionKeywords = [
        "please",
        "can you",
        "could you",
        "reply",
        "confirm",
        "review",
        "submit",
        "send",
        "update",
        "complete",
        "bring",
        "investigate",
        "schedule",
        "collect",
        "resend",
        "check",
        "countersign",
        "shortlisting",
        "shortlist",
        "respond",
        "sign"
    ]

    # If any action keyword appears, predict true
    for keyword in actionKeywords:
        if keyword in cleanedMessageText:
            return "true"

    # Otherwise predict false
    return "false"


def predictPriorityLabel(messageText):
    # Convert the message into normalised lowercase text
    cleanedMessageText = normaliseTextValue(messageText)

    # Strong urgency signals
    urgentKeywords = [
        "immediately",
        "as soon as possible",
        "today",
        "right now",
        "by close of business today",
        "by end of day",
        "before 4pm",
        "before 5pm",
        "in 48 hours",
        "midnight",
        "critical",
        "affecting live transactions",
        "as soon as possible",
        "now",
        "overdue",
        "urgent",
        "escalate immediately",
        "help needed urgently"

    ]

    # Important but not fully urgent signals
    importantKeywords = [
        "by friday",
        "by tomorrow",
        "before tomorrow",
        "next week",
        "review",
        "confirm your attendance",
        "please confirm",
        "deadline",
        "expires",
        "due",
        "prepare",
        "risk register",
        "schedule it in",
        "respond",
        "by thursday",
        "by wednesday",
        "by tuesday",
        "by sunday",
        "by saturday",
        "by monday",
        "important",
        "priority",
        "escalate",
        "escalation"
    ]

    # Informational patterns
    informationalKeywords = [
        "fyi",
        "no action needed",
        "just letting you know",
        "more details to follow",
        "completed successfully",
        "can be found on the intranet",
        "has been installed",
        "will be turned off briefly",
        "has been postponed",
        "has been changed",
        "will be resurfaced",
        "congratulations",
        "is now available",
        "have been placed",
        "please be aware",
        "please note",
        "for your information"
    ]

    # Check urgent first
    for keyword in urgentKeywords:
        if keyword in cleanedMessageText:
            return "urgent"

    # Then check important
    for keyword in importantKeywords:
        if keyword in cleanedMessageText:
            return "important"

    # Then check informational
    for keyword in informationalKeywords:
        if keyword in cleanedMessageText:
            return "informational"

    # Default fallback
    return "routine"


def runBaseline(inputFilePath, outputFilePath):
    # Load the test dataset
    dataFrame = pd.read_csv(inputFilePath)

    # Create prediction columns
    predictedPriorityLabels = []
    predictedActionValues = []

    # Predict each row one at a time
    for messageText in dataFrame["message_text"]:
        predictedPriorityLabel = predictPriorityLabel(messageText)
        predictedActionValue = predictActionRequired(messageText)

        predictedPriorityLabels.append(predictedPriorityLabel)
        predictedActionValues.append(predictedActionValue)

    # Store predictions in new columns
    dataFrame["predicted_priority_label"] = predictedPriorityLabels
    dataFrame["predicted_action_required"] = predictedActionValues

    # Save prediction file
    dataFrame.to_csv(outputFilePath, index=False)

    return dataFrame


if __name__ == "__main__":
    inputFilePath = "data/processed/test.csv"
    outputFilePath = "outputs/predictions/baseline_predictions.csv"

    predictionDataFrame = runBaseline(inputFilePath, outputFilePath)

    print(f"Baseline predictions saved to: {outputFilePath}")
    print(f"Rows predicted: {len(predictionDataFrame)}")

    print("\nPrediction preview:")
    print(
        predictionDataFrame[
            [
                "message_id",
                "message_text",
                "priority_label",
                "predicted_priority_label",
                "action_required",
                "predicted_action_required"
            ]
        ].head()
    )