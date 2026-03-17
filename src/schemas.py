# Allowed values for the main priority classification task
ALLOWED_PRIORITY_LABELS = [
    "urgent",
    "important",
    "routine",
    "informational"
]

# Allowed source types for dataset rows
ALLOWED_SOURCE_LABELS = [
    "synthetic",
    "realistic_synthetic",
    "adapted_real_world"
]

# Allowed dataset split values
ALLOWED_SPLIT_LABELS = [
    "train",
    "test"
]

# Allowed values for the action_required column
ALLOWED_ACTION_REQUIRED_VALUES = [
    "true",
    "false"
]

# Required columns that must exist in the dataset
REQUIRED_DATASET_COLUMNS = [
    "message_id",
    "source",
    "message_text",
    "priority_label",
    "action_required"
]