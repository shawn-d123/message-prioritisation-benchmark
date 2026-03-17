import pandas as pd
from sklearn.model_selection import train_test_split


def splitDataset(inputFilePath, trainFilePath, testFilePath):
    # Load the full processed dataset
    dataFrame = pd.read_csv(inputFilePath)

    # Split the dataset into training and test sets
    # stratify=dataFrame["priority_label"] helps keep class balance similar in both splits
    trainDataFrame, testDataFrame = train_test_split(
        dataFrame,
        test_size=0.25,
        random_state=42,
        stratify=dataFrame["priority_label"]
    )

    # Add a split column so each file clearly shows what it is
    trainDataFrame["split"] = "train"
    testDataFrame["split"] = "test"

    # Save both files
    trainDataFrame.to_csv(trainFilePath, index=False)
    testDataFrame.to_csv(testFilePath, index=False)

    return trainDataFrame, testDataFrame


if __name__ == "__main__":
    inputFilePath = "data/processed/priority_dataset.csv"
    trainFilePath = "data/processed/train.csv"
    testFilePath = "data/processed/test.csv"

    trainDataFrame, testDataFrame = splitDataset(
        inputFilePath,
        trainFilePath,
        testFilePath
    )

    print(f"Train set saved to {trainFilePath}")
    print(f"Test set saved to {testFilePath}")
    print(f"Training rows: {len(trainDataFrame)}")
    print(f"Test rows: {len(testDataFrame)}")

    print("\nTraining class counts:")
    print(trainDataFrame["priority_label"].value_counts())

    print("\nTest class counts:")
    print(testDataFrame["priority_label"].value_counts())