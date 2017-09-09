import pandas as pd
import numpy as np
import random as rnd
import tensorflow as tf

TRAIN_CSV_FILE = '~/Downloads/train.csv'
TEST_CSV_FILE = '~/Downloads/test.csv'

train_df = pd.read_csv(TRAIN_CSV_FILE)
test_df = pd.read_csv(TEST_CSV_FILE)
combine = [train_df, test_df]

# Create Title feature from Name
for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Shrink Title space
for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

# Convert Title to numerical feature
for dataset in combine:
	dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
	dataset['Title'] = dataset['Title'].fillna(0)

# Now we can drop Name and PassengerId
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop('Name', axis=1)
combine = [train_df, test_df]

# Convert Sex to numerical feature
for dataset in combine:
	dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)

# There are some null ages. And we know there's a correlation between (Pclass, Sex) and age.
# So let's generate guesses based on the medians of the two.
guess_ages = np.zeros((2, 3))

for dataset in combine:
	for i in range(0, 2):
		for j in range(0, 3):
			guess_df = dataset.loc[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
			age_guess = guess_df.median()

			guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

	for i in range(0, 2):
		for j in range(0, 3):
			dataset.loc[(dataset.Age.isnull()) & (dataset['Sex'] == i) & (dataset['Pclass'] == j + 1), 'Age'] = guess_ages[i,j]

	dataset['Age'] = dataset['Age'].astype(int)

# Convert ages into bands. These band values were found experimenting with pd.cut
for dataset in combine:
	dataset.loc[(dataset.Age <= 16), 'Age'] = 0
	dataset.loc[(dataset.Age > 16) & (dataset.Age <= 32), 'Age'] = 1
	dataset.loc[(dataset.Age > 32) & (dataset.Age <= 48), 'Age'] = 2
	dataset.loc[(dataset.Age > 48) & (dataset.Age <= 64), 'Age'] = 3
	dataset.loc[(dataset.Age > 64), 'Age'] = 4

# Generate IsAlone
for dataset in combine:
	dataset.loc[(dataset.Parch + dataset.SibSp == 0), 'IsAlone'] = 1
	dataset.loc[(dataset.Parch + dataset.SibSp > 0), 'IsAlone'] = 0

# Drop useless columns
train_df = train_df.drop(['Parch', 'SibSp', 'Cabin', 'Ticket'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'Cabin', 'Ticket'], axis=1)
combine = [train_df, test_df]

# Convert Fare into bands. These band values were found experimenting with pd.qcut
for dataset in combine:
	dataset.loc[(dataset.Fare <= 7.854), 'Fare'] = 0
	dataset.loc[(dataset.Fare > 7.854) & (dataset.Fare <= 10.5), 'Fare'] = 1
	dataset.loc[(dataset.Fare > 10.5) & (dataset.Fare <= 21.679), 'Fare'] = 2
	dataset.loc[(dataset.Fare > 21.679) & (dataset.Fare <= 39.688), 'Fare'] = 3
	dataset.loc[(dataset.Fare > 39.688), 'Fare'] = 4

# Fill embarked missing values with most frequent result, then map to numerical values
most_freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
	dataset['Embarked'] = dataset.Embarked.fillna(most_freq_port)
	dataset['Embarked'] = dataset.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 7 numerical features
feature_columns = [tf.feature_column.numeric_column("x", shape=[7])]

# For now, I'm experimenting using the same number of layers and neurons as in the TF intro guide to estimators.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=2,
                                        model_dir="/tmp/titanic_model")

# Convert to ndarrays
train_x = train_df.drop('Survived', axis=1).values
train_y = train_df['Survived'].values

# Define input function
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_x},
    y=train_y,
    num_epochs=None,
    shuffle=True)

# Train!
classifier.train(input_fn=train_input_fn, steps=2000)

# Now we make predictions using text_x
test_x = test_df.drop('PassengerId', axis=1).values
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_x_copy},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"][0] for p in predictions]

submission = pd.DataFrame({
	"PassengerId": test_df['PassengerId'],
	"Survived": predicted_classes,	
})

# This resulted in 78.947% accuracy.
submission.to_csv('submission.csv', index=False)