import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def loadDatasets(trainFilePath, testFilePath):
    # Load the training and test datasets
    trainDataFrame = pd.read_csv(trainFilePath)
    testDataFrame = pd.read_csv(testFilePath)

    return trainDataFrame, testDataFrame


def buildVectorizer():
    # Create a TF-IDF vectorizer to convert text into numeric features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    return vectorizer


def trainPriorityModels(trainMessages, trainPriorityLabels):
    # Create and train the priority classification models
    priorityNaiveBayesModel = MultinomialNB()

    priorityLogisticModel = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    priorityNaiveBayesModel.fit(trainMessages, trainPriorityLabels)
    priorityLogisticModel.fit(trainMessages, trainPriorityLabels)

    return priorityNaiveBayesModel, priorityLogisticModel


def trainActionModels(trainMessages, trainActionLabels):
    # Create and train the binary action_required models
    actionNaiveBayesModel = MultinomialNB()

    actionLogisticModel = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    actionNaiveBayesModel.fit(trainMessages, trainActionLabels)
    actionLogisticModel.fit(trainMessages, trainActionLabels)

    return actionNaiveBayesModel, actionLogisticModel


def savePredictionFile(
    outputFilePath,
    testDataFrame,
    predictedPriorityLabels,
    predictedActionLabels
):
    # Copy the test dataframe so we do not accidentally modify the original
    predictionDataFrame = testDataFrame.copy()

    # Add prediction columns
    predictionDataFrame["predicted_priority_label"] = predictedPriorityLabels
    predictionDataFrame["predicted_action_required"] = predictedActionLabels

    # Save prediction results
    predictionDataFrame.to_csv(outputFilePath, index=False)

    return predictionDataFrame


def trainAndPredict(trainFilePath, testFilePath):
    # Load the datasets
    trainDataFrame, testDataFrame = loadDatasets(trainFilePath, testFilePath)

    # Create the vectorizer
    vectorizer = buildVectorizer()

    # Learn the vocabulary from training text only so the model does not see the test set in advance.
    trainMessageFeatures = vectorizer.fit_transform(trainDataFrame["message_text"])

    # Transform test text using the same learned vocabulary
    testMessageFeatures = vectorizer.transform(testDataFrame["message_text"])

    # Train priority models
    priorityNaiveBayesModel, priorityLogisticModel = trainPriorityModels(
        trainMessageFeatures,
        trainDataFrame["priority_label"]
    )

    # Train action_required models
    actionNaiveBayesModel, actionLogisticModel = trainActionModels(
        trainMessageFeatures,
        trainDataFrame["action_required"].astype(str).str.strip().str.lower()
    )

    # Predict with Naive Bayes
    nbPredictedPriorityLabels = priorityNaiveBayesModel.predict(testMessageFeatures)
    nbPredictedActionLabels = actionNaiveBayesModel.predict(testMessageFeatures)

    # Predict with Logistic Regression
    logisticPredictedPriorityLabels = priorityLogisticModel.predict(testMessageFeatures)
    logisticPredictedActionLabels = actionLogisticModel.predict(testMessageFeatures)

    # Save both prediction files
    nbPredictionDataFrame = savePredictionFile(
        "outputs/predictions/nb_predictions.csv",
        testDataFrame,
        nbPredictedPriorityLabels,
        nbPredictedActionLabels
    )

    logisticPredictionDataFrame = savePredictionFile(
        "outputs/predictions/logistic_predictions.csv",
        testDataFrame,
        logisticPredictedPriorityLabels,
        logisticPredictedActionLabels
    )

    return nbPredictionDataFrame, logisticPredictionDataFrame


if __name__ == "__main__":
    trainFilePath = "data/processed/train.csv"
    testFilePath = "data/processed/test.csv"

    nbPredictionDataFrame, logisticPredictionDataFrame = trainAndPredict(
        trainFilePath,
        testFilePath
    )

    print("Naive Bayes predictions saved to: outputs/predictions/nb_predictions.csv")
    print("Logistic Regression predictions saved to: outputs/predictions/logistic_predictions.csv")

    print("\nNaive Bayes preview:")
    print(
        nbPredictionDataFrame[
            [
                "message_id",
                "priority_label",
                "predicted_priority_label",
                "action_required",
                "predicted_action_required"
            ]
        ].head()
    )

    print("\nLogistic Regression preview:")
    print(
        logisticPredictionDataFrame[
            [
                "message_id",
                "priority_label",
                "predicted_priority_label",
                "action_required",
                "predicted_action_required"
            ]
        ].head()
    )