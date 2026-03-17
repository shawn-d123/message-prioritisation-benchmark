import pandas as pd
from schemas import (
    ALLOWED_PRIORITY_LABELS,
    ALLOWED_SOURCE_LABELS,
    ALLOWED_ACTION_REQUIRED_VALUES,
    REQUIRED_DATASET_COLUMNS
)


def validateDataset(datasetFilePath):
    validationErrors = [] # stores error messages found during validation

    dataFrame = pd.read_csv(datasetFilePath)

    # Check 1: required columns exist
    for columnName in REQUIRED_DATASET_COLUMNS:
        if columnName not in dataFrame.columns:
            validationErrors.append(f"Missing required column: {columnName}")

    # Stop early if columns are missing, because later checks depend on them
    if len(validationErrors) > 0:
        return validationErrors

    # Check 2: priority_label values are valid
    for priorityLabel in dataFrame["priority_label"]:
        if priorityLabel not in ALLOWED_PRIORITY_LABELS:
            validationErrors.append(f"Invalid priority label found: {priorityLabel}")

    # Check 3: source values are valid
    for sourceValue in dataFrame["source"]:
        if sourceValue not in ALLOWED_SOURCE_LABELS:
            validationErrors.append(f"Invalid source value found: {sourceValue}")

    # Check 4: action_required values are valid
    for actionValue in dataFrame["action_required"]:
        if actionValue not in ALLOWED_ACTION_REQUIRED_VALUES:
            validationErrors.append(f"Invalid action_required value found: {actionValue}")

    # Check 5: duplicate message IDs
    duplicateIds = dataFrame[dataFrame["message_id"].duplicated()]["message_id"].tolist()
    for duplicateId in duplicateIds:
        validationErrors.append(f"Duplicate message_id found: {duplicateId}")

    # Check 6: empty message text
    for rowNumber, messageText in enumerate(dataFrame["message_text"], start=1):
        if pd.isna(messageText) or str(messageText).strip() == "":
            validationErrors.append(f"Empty message_text found in row {rowNumber}")

    return validationErrors


if __name__ == "__main__":
    datasetFilePath = "data/processed/priority_dataset.csv"
    validationErrors = validateDataset(datasetFilePath)

    if len(validationErrors) == 0:
        print("Dataset validation passed.")
    else:
        print("Dataset validation failed.")
        print("Problems found:")
        for errorMessage in validationErrors:
            print(f"- {errorMessage}")