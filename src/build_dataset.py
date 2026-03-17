import pandas as pd
from schemas import REQUIRED_DATASET_COLUMNS


def buildDataset(inputFilePath, outputFilePath):
    # Load the raw dataset that was generated externally
    dataFrame = pd.read_csv(inputFilePath)

    # Clean the column names in case there are accidental spaces
    dataFrame.columns = [str(columnName).strip() for columnName in dataFrame.columns]

    # Keep only the columns that belong in the benchmark schema
    dataFrame = dataFrame[REQUIRED_DATASET_COLUMNS]

    # Clean text values so the dataset is more consistent
    dataFrame["message_id"] = dataFrame["message_id"].astype(str).str.strip()
    dataFrame["source"] = dataFrame["source"].astype(str).str.strip().str.lower()
    dataFrame["message_text"] = dataFrame["message_text"].astype(str).str.strip()
    dataFrame["priority_label"] = dataFrame["priority_label"].astype(str).str.strip().str.lower()
    dataFrame["action_required"] = dataFrame["action_required"].astype(str).str.strip().str.lower()

    # Save the cleaned dataset into the processed folder
    dataFrame.to_csv(outputFilePath, index=False)

    return dataFrame


if __name__ == "__main__":
    inputFilePath = "data/raw/generated_dataset.csv"
    outputFilePath = "data/processed/priority_dataset.csv"

    dataFrame = buildDataset(inputFilePath, outputFilePath)

    print(f"Processed dataset saved to: {outputFilePath}")
    print(f"Total rows: {len(dataFrame)}")

    print("\nPriority label counts:")
    print(dataFrame["priority_label"].value_counts())

    print("\nAction required counts:")
    print(dataFrame["action_required"].value_counts())