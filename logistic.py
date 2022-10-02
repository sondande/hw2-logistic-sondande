"""
Assignment #2: Logistic Regression

By Sagana Ondande & Ada Ates

The assignment works on the following main goals:
1. Implement logistic regression as a valuable method to supervised machine learning (and an eventual building block for neural networks),
2. Practice with data pre-processing to prepare datasets for supervised learning,
3. Investigate the learning process during training.
4. Practice working with a partner on code development and scientific experimentation.
"""

# Import libraries
import os
import sys
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy.special import expit
from sklearn import metrics
"""
Takes the following parameters:

a. The path to a file containing a data set (e.g., monks1.csv)
b. The learning rate 𝜂 to use during stochastic gradient descent
c. The percentage of instances to use for a training set
d. The percentage of instances to use for a validation set
e. A random seed as an integer
"""

def data_preprocessing(dataset):
    # Determine whether a column contains numerical or nominial values
    # Create a new Pandas dataframe to maintain order of columns when doing One-Hot Coding on Nominial values
    new_dataframe = pd.DataFrame()
    # Iterate through all the columns of the training_set 
    for x in dataset.columns:
        # Determine if the column 'x' in training set is a Nominial Data or Numerical 
        if is_string_dtype(dataset[x]) and not is_numeric_dtype(dataset[x]):
            # Apply One-Hot Encoding onto Pandas Series at column 'x' 
            dummies = pd.get_dummies(dataset[x], prefix=x, prefix_sep='.', drop_first=True)
            # Combine the One-Hot Encoding Dataframe to our new dataframe to the new_dataframe 
            new_dataframe = pd.concat([new_dataframe, dummies],axis=1)
        else: 
            # Find the maximum value in column 'x'
            max_value = max(dataset[x])
            # Find the minimum value in column 'x'
            min_value = min(dataset[x])
            # If the max and min aren't 0 to ensure that we don't do zero division
            if max_value != 0 and min_value != 0:
                # Apply net value formula to every value in pandas dataframe
                dataset[x] = dataset[x].apply(lambda y: (y - min_value)/(max_value - min_value))
            # Combine New column to our new_dataframe
            new_dataframe = pd.concat([new_dataframe, dataset[x]],axis=1)
    return new_dataframe

"""
Creation of a Sigmoid Function that handles overflow cases as well
"""
def sigmoid(net):
    # If x is a very large positive number, the sigmoid function will be close to 1
    if net >= 0:
        temp = np.exp(-net)
        out = 1 / (1 + temp)
    # If x is a very large negative number, the sigmoid function will be close to 0
    else:
        temp = np.exp(net)
        out = temp / (1 + temp)
    return out
    # return expit(net)

"""
Calculate Net Value for Stochastic Gradient Descent
"""
def net_calculate(weights, x_instance):
    # take the first value from weights as it is part of the net without a corresponding value in the instance
    net = weights[0]
    # Instances and weights are the same length
    for i in range(1, len(x_instance)):
        net += weights[i] * x_instance[i]
    return net


def stochastic_gradient_descent(np_ts, np_vs, learning_rate):
    # 1. Choose random values for all weights (often between -0.1 and 0.1)
    weights = np.random.uniform(-0.1, 0.1, training_set.shape[1])
    # 2. Until either the accuracy on the validation set > A% or we run n epochs
    # Set accuracy variable 
    accuracy = 0
    epochs = 0
    while accuracy <= 0.99:
        if epochs > 500:
            break
        # A. For each instance x in the training set
        for insta_count in range(len(np_ts)):
            instance_label = np_ts[insta_count][0]
            # Calculate Net Value between X instance and weights
            net = net_calculate(weights=weights, x_instance=np_ts[insta_count])
            # Calculate the out values from the net values calculated above 
            out_value = sigmoid(net=net)
            # I. Calculate gradient of w0
            grad_w0 = 0
            grad_w0 = -1 * out_value * (1 - out_value) * (instance_label - out_value)
            # Update first weight in weights
            weights[0] -= (learning_rate * grad_w0)
            # II. Calculate gradient of wi
            for attr_count in range(1, len(np_ts[0])):
                grad_wi = 0
                grad_wi = -1 * np_ts[insta_count][attr_count] * out_value * (1 - out_value) * (instance_label - out_value)
                weights[attr_count] -= (learning_rate * grad_wi)
            # print(f"Updated weights list: {weights}\n")
        epochs += 1
        tt = 0
        tf = 0 
        ft = 0
        ff = 0
        # Testing against validation set
        for insta_count in range(len(np_vs)):
            instance_label = np_vs[insta_count][0]
            # Calculate Net Value between X instance and weights
            net = net_calculate(weights=weights, x_instance=np_vs[insta_count])
            # Calculate the out values from the net values calculated above 
            out_value = sigmoid(net=net)
            predict = 1 if out_value > 0.5 else 0
            # print(f"Predict Value:{predict}")
            if predict == 1 and instance_label == 1:
                tt += 1 
            elif predict == 1 and instance_label == 0:
                ft += 1 
            elif predict == 0 and instance_label == 1:
                tf += 1 
            else: 
                ff += 1
        accuracy = (tt + ff) / (tt + tf + ft + ff)
    print(f"Completed Epoch:{epochs}\nAccuracy: {accuracy}\nWeights: {weights}\n")

    # outFileName = f"results-{os.path.basename(file_path)}-{learning_rate}-{randomSeed}.csv"
    # # Writing Seciton
    # data_input = {"0": [ft, tf], "1"}
    # confusion_matrix
    # outputFile = open("log-results/"+outFileName, 'w')

    # writer = csv.writer(outputFile)
    # # Write Labels Row
    # writer.writerow(possible_labels)

    return weights

def model(dataset, weight):
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    actualLabel = []
    predictLabels = []
    filename = ("results-" + str(file_path) + "-" + str(learning_rate) + "r" + "-" + str(randomSeed) + ".csv")
    for insta_count in range(len(dataset)):
        actualLabel.append(dataset[insta_count][0])
        # Calculate Net Value between X instance and weights
        net = net_calculate(weights=weight, x_instance=dataset[insta_count])
        # Calculate the out values from the net values calculated above
        out_value = sigmoid(net=net)
        predict = 1 if out_value > 0.5 else 0
        predictLabels.append(predict)
        if predict == 1 and dataset[insta_count][0] == 1:
            tt += 1
        elif predict == 1 and dataset[insta_count][0] == 0:
            tf += 1
        elif predict == 0 and dataset[insta_count][0] == 1:
            ft += 1
        else:
            ff += 1
        accuracy = (tt + ff) / (tt + tf + ft + ff)
    print(f"Accuracy: {accuracy}\nWeights: {weight}\n")

    labels = np.unique(actualLabel)

    #print("confusion matrix from metrics:")
    #cm = metrics.confusion_matrix(actualLabel, predictLabels)
    #print(cm)
    #df = pd.DataFrame(cm, index=labels, columns=labels)
    #for row in df.values:
        #print(str(tuple(row))[1:-1])

    size = len(actualLabel)
    matrix = dict()

    # create matrix initialised with 0 (nested dictionary)
    for class_name in labels:
        matrix[class_name] = {label: 0 for label in labels}

    # form the confusion matrix by incrementing proper places
    for i in range(size):
        actual_class = actualLabel[i]
        # print("actual_class: ", actual_class)
        pred_class = predictLabels[i]
        # print("pred_class:", pred_class)
        matrix[actual_class][pred_class] += 1
        #print("matrix: ", matrix[actual_class][pred_class])

    matrix = dict(zip(labels, list(matrix.values())))

    print("Confusion Matrix of given model is :")
    print("Predicted Label")
    keys = list(matrix.keys())
    print(",".join(str(e) for e in keys))
    for key, value in matrix.items():
        for pred, count in value.items():
            #print("key, value", key, value)
            print(count, end=",")  # counts in predictLabel & true matching or false counts
        print("%s" % key)  # respective keys
    # print("true, pred: ", true, predictLabel)      # test-related print statement

    with open(filename, "w") as f:
        f.write((",".join(str(e) for e in keys)))
        f.write('\n')
        for key, value in matrix.items():
            for pred, count in value.items():
                f.write(str(count))  # counts in predictLabel & true matching or false counts
                f.write(",")
            f.write("%s" % key)  # respective keys
            f.write("\n")

    return accuracy

# Beginning of code
try:
    # Get Dataset File
    # a.The path to a file containing a data set (e.g., monks1.csv)
    file_path = sys.argv[1]

    # b. The learning rate 𝜂 to use during stochastic gradient descent
    learning_rate = float(sys.argv[2])

    #c. The percentage of instances to use for a training set
    training_set_percent = float(sys.argv[3])
    # Ensure training set percent is a valid percent that can be used
    if 0 > training_set_percent or training_set_percent > 1:
        print("Invalid percent. Please choose a value between 0 and 1")
        exit(1)

    #d. The percentage of instances to use for a validation set
    validation_set_percent = float(sys.argv[4])

    # Ensure validation set percent is a valid percent that can be used
    if 0 > validation_set_percent or validation_set_percent > 1:
        print("Invalid percent. Please choose a value between 0 and 1")
        exit(1)

    # Check that the values don't exceed 100%
    if training_set_percent + validation_set_percent == 1:
        print("Fair warning ... you don't have a testing set...\nPlease try again and leave room for a testing set :)")
        exit(1)
    elif training_set_percent + validation_set_percent > 1:
        print(f"The percentage of the training set plus the validation set is equal to: {training_set_percent + validation_set_percent}\nPlease only input values who's sum is less than 1")
        exit(1)

    # Store the size of the testing set
    testing_set_percent = 1 - training_set_percent - validation_set_percent

    #e. A random seed as an integer
    randomSeed = int(sys.argv[5])

    # Print all input values given for user to see
    print(f"Inputs:\nFile: {file_path}\nLearning rate: {learning_rate}")
    print(f"Training Set Percent: {training_set_percent}\nValidation Set Percent: {validation_set_percent}\nTesting Set Percent: {testing_set_percent}")
    print(f"Random Seed: {randomSeed}\n")

    # Read in dataset
    df = pd.read_csv(file_path)
    #labels = df.iloc[:, 0]

    # shuffle the dataframe. Use random seed from input and fraction 1 as we want the whole dataframe
    shuffled_df = df.sample(frac=1,random_state=randomSeed)

    print(f"Number of Instances in Dataframe: {len(df)}")

    # Applies the splits to the training set and validation set. The last argument wil what remained whihc is the testing set
    # Note, Numpy split does it where there are equal parts. First one says take the first <test_set_percent> amount of my dataframe
    # Second says, Take <training_set_percent + validation_set_percent> as training_set_percent is already taken so that leaves just the validation set amount
    # These points go by indices so that index of where to start the validation set is the sum of the two. The remaining amount is the left over argument that is the part of the dataframe not taken.
    # This results in that being the testing set
    splits_indices = [int(training_set_percent * len(df)), int((training_set_percent + validation_set_percent) * len(df))]
    print(f"Splits indexes they begin at: {splits_indices}\n")
    training_set, validation_set, testing_set = np.split(shuffled_df, splits_indices)

    # Print out the lengths of the training, validation, and testing sets
    print(f"Length of training: {len(training_set)}")
    print(f"Length of validiation set: {len(validation_set)}")
    print(f"Length of testing: {len(testing_set)}\n")

    # Preprocess the data
    training_set = data_preprocessing(training_set).to_numpy()
    validation_set = data_preprocessing(validation_set).to_numpy()
    testing_set = data_preprocessing(testing_set).to_numpy()

    # Train the model
    print("Beginning Training")
    weights = stochastic_gradient_descent(training_set, validation_set, learning_rate)
    print("Training Complete")
    model(testing_set, weights)

except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)

