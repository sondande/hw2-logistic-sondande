"""
Assignment #2: Logistic Regression

By Sagana Ondande

The assignment works on the following main goals:
1. Implement logistic regression as a valuable method to supervised machine learning (and an eventual building block for neural networks),
2. Practice with data pre-processing to prepare datasets for supervised learning,
3. Investigate the learning process during training.
4. Practice working with a partner on code development and scientific experimentation.
"""

# Import libraries
import sys
from xxlimited import new
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

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
            dummies = pd.get_dummies(dataset[x], prefix=x, prefix_sep='.')
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
                dataset[x] = dataset[x].apply(lambda x: (x - min_value)/(max_value - min_value))
            # Combine New column to our new_dataframe
            new_dataframe = pd.concat([new_dataframe, dataset[x]],axis=1)
    return new_dataframe

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

    print(f"Length of training: {len(training_set)}")
    print(f"Length of validiation set: {len(validation_set)}")
    print(f"Length of testing: {len(testing_set)}")

    training_set = data_preprocessing(training_set)
    validation_set = data_preprocessing(validation_set)
    testing_set = data_preprocessing(testing_set)

    print(training_set)
    
except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)

