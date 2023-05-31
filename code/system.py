"""Letter Classification and Word Search System

Solution the COM2004/3004 assignment.

Code written by Lucy Lau (and some from the CoCalc Labs)

"""

from typing import List

import numpy as np
from utils import utils
from utils.utils import Puzzle

import scipy.linalg

np.random.seed(0)

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20

def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Uses Principal Component Analysis to reduce dimensionality of feature vectors.

    Code from the CoCalc Lab 7 was used.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    # Finding Principal Components (code from CoCalc Lab 7)
    covData = np.cov(model["training_data"], rowvar=0)
    N = covData.shape[0]
    numComponents = 20
    w, v = scipy.linalg.eigh(covData, eigvals=(N - numComponents, N - 1))
    v = np.fliplr(v)

    pca_data = np.dot((data - model["training_mean"]), v)

    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.
    
    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}

    # Stores the labels of the training data
    model["labels_train"] = labels_train.tolist()

    # Stores the raw training data
    model["training_data"] = fvectors_train.tolist()

    mean = np.mean(fvectors_train)

    # Stores the mean of the training data
    model["training_mean"] = mean

    # Stores the reduced training data
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model



def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Letter classifier.

    Uses a KNN algorithm using cosine distance to classify squares, with k = 10

    Args:
        fvectors_test (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """

    train = np.array(model["fvectors_train"])
    train_labels = np.array(model["labels_train"])
    test = fvectors_test

    # Using cosine distance (from CoCalc Labs)
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())

    # KNN: 10 nearest neighbours
    k = 10
    knearest = (np.argsort(-dist, axis=1))[:, :k]

    knearest_labels = []

    # Matches index with the label
    for row in knearest:
        temp = []
        for index in row:
            letter_label = train_labels[index]
            temp.append(letter_label)
        knearest_labels.append(temp)

    # Returns the most common label
    nearest = []
    for row in knearest_labels:
        l = max(set(row), key=row.count)
        nearest.append(l)

    return nearest

def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Word search algorithm.

    For every word, it searches for it in the letter grid and returns its position
    in the form (startLetter, endLetter)

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """

    results = []

    for word in words:
        # Capitalises word so consistent with letter grid
        word = word.upper()
        possiblePosition = ((0,0,0,0), -1)

        for row in range(len(labels)):
            for column in range(len(labels[0])):

                # Searches every direction and every start position for the word in the grid
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        direction = (i, j) # Vector direction

                        if direction != (0, 0):
                            # Creates a new word by following a direction in the grid
                            newWord, position = createWord((row, column), direction, word, labels)
                            newWordJoin = "".join(newWord)

                            if len(newWordJoin) == len(word):
                                # Check how many letters are correct
                                correct = 0
                                for k in range(len(word)):
                                    if newWordJoin[k] == word[k]:
                                        correct += 1
                                # If there are more correct letters in the new word then replace possiblePosition
                                if correct > possiblePosition[1]:
                                    possiblePosition = (position, correct)

        # Add to result
        results.append(possiblePosition[0])

    return results

# Creates a word by following a direction, returns the word created
# and its position in the letter grid
def createWord(start, direction, word, labels):

    positions = [start]
    output = [labels[start]]
    currentPos = start
    
    for i in range(len(word) - 1):
        currentPos = (currentPos[0] + direction[0], currentPos[1] + direction[1])

        if 0 <= currentPos[0] < len(labels) and 0 <= currentPos[1] < len(labels[0]):
            output.append(labels[currentPos])
            positions.append(currentPos)

    positions = positions[0] + positions[-1]
    
    return output, positions

