import numpy as np

# Step 1: Initialize M and B matrices
M = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10],
              [11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25]])

B = np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]])

# Step 2: Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 3: Matrix multiplication with NumPy
def matrix_multiplication_numpy(X):
    return np.dot(M, X) + B

def matrix_multiplication_manual(X):
    result = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):  # Loop through rows of M
        for j in range(5):  # Loop through columns of X
            for k in range(5):  # Element-wise multiplication and sum
                result[i][j] += M[i][k] * X[k][j]
    # Add bias B
    for i in range(5):
        for j in range(5):
            result[i][j] += B[i][j]
    return result






from fastapi import FastAPI
import uvicorn 
import numpy as np 

app = FastAPI()



# sigmoid function
def sigmoid(x):
    """
    This sigmoid funnctions returns an 
    output between 0 and 1.. 

    the formula is 1/ (1 + e(-x))
    but e is equivalent to 2.71828
    """
    
    return 1 / (1 + 2.71828 ** (-x) )


# class MatrixInput(BaseModel):
#     """
#     this model validates the input going into the function
#     making sure they're matrix of list[list] with floats
#     """
#     matrix : list[list[float]]



# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
def f(x):
    pass
 
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy

def matrix_multiplication_numpy(X):
    return np.dot(M, X) + B

def matrix_multiplication_manual(X):
    result = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):  # Loop through rows of M
        for j in range(5):  # Loop through columns of X
            for k in range(5):  # Element-wise multiplication and sum
                result[i][j] += M[i][k] * X[k][j]
    # Add bias B
    for i in range(5):
        for j in range(5):
            result[i][j] += B[i][j]
    return result
#Return 

#initialize x as a 5 * 5 matrix
X = np.array([[1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]])

#Make a call to the function
# Perform (M * X) + B using NumPy
result = matrix_multiplication_numpy(X)
print("\nMatrix Multiplication Result (M * X + B):")
print(result)

# Perform (M * X) + B manually
manual_result = matrix_multiplication_manual(X)
print("\nManual Matrix Multiplication Result (M * X + B):")
print(manual_result)

#Recreate the function with the sigmoid Function
sigmoid_result = sigmoid(result)
print("\nSigmoid Result:")
print(sigmoid_result)

if __name__ == "__main__":
    
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

