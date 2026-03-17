import pandas as pd
from schemas import REQUIRED_DATASET_COLUMNS


def buildDatasetRows():
    # Create a list of dictionaries.
    # Each dictionary represents one dataset row.
    datasetRows = [
        {
            "message_id": "msg_001",
            "source": "synthetic",
            "message_text": "Please send the signed form by 5pm today.",
            "priority_label": "urgent",
            "action_required": "true"
        },
        {
            "message_id": "msg_002",
            "source": "synthetic",
            "message_text": "FYI the office will be closed next Monday.",
            "priority_label": "informational",
            "action_required": "false"
        },
        {
            "message_id": "msg_003",
            "source": "realistic_synthetic",
            "message_text": "Can you confirm your attendance by Friday?",
            "priority_label": "important",
            "action_required": "true"
        },
        {
            "message_id": "msg_004",
            "source": "synthetic",
            "message_text": "Reminder to bring your ID badge tomorrow.",
            "priority_label": "routine",
            "action_required": "true"
        },
        {
            "message_id": "msg_005",
            "source": "synthetic",
            "message_text": "Post the weekly report in the shared folder by end of day.",
            "priority_label": "urgent",
            "action_required": "true"
        },
        {
            "message_id": "msg_006",
            "source": "synthetic",
            "message_text": "Don't forget to submit your timesheet by Friday.",
            "priority_label": "important",
            "action_required": "true"
        },
        {
            "message_id": "msg_007",
            "source": "synthetic",
            "message_text": "Reminder to collect your new ID badge from reception every Monday.",
            "priority_label": "routine",
            "action_required": "true"
        },
        {
            "message_id": "msg_008",
            "source": "synthetic",
            "message_text": "There may be a delay in response due to high volume of requests.",
            "priority_label": "informational",
            "action_required": "false"
        }
    ]

    return datasetRows


def saveDataset(datasetRows, outputFilePath):
    dataFrame = pd.DataFrame(datasetRows)

    dataFrame = dataFrame[REQUIRED_DATASET_COLUMNS]

    dataFrame.to_csv(outputFilePath, index=False)


if __name__ == "__main__":
    outputFilePath = "data/processed/priority_dataset.csv"
    datasetRows = buildDatasetRows()
    saveDataset(datasetRows, outputFilePath)

    print(f"Dataset saved to {outputFilePath}")
    print(f"Total rows: {len(datasetRows)}")