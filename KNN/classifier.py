from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def trainKNN(data, labels, neighbors, jobs=-1):
    le = LabelEncoder()

    labels = le.fit_transform(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs)
    model.fit(trainX, trainY)

    print(classification_report(testY, model.predict(testX), target_names=le.classes_))

    return model
