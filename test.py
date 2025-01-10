from fastapi import FastAPI
import uvicorn 
import numpy as np 
from pydantic import BaseModel


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


class MatrixInput(BaseModel):
    """
    this model validates the input going into the function
    making sure they're matrix of list[list] with floats
    """
    matrix : list[list[float]]

#initialize M and B
M = np.ones((5, 5))
B = np.zeros((5, 5))

# use the post decorator directly below this
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy
@app.post("/calculate")
def matrix_multiplication(input_data: MatrixInput, method: str = 'numpy'):
    '''
        We will perform the multiplication of MX + B using the method provided
        
        - if method == "numpy" we calculate with numpy, else we use manual
        - Default method is "Numpy"
    '''

    X = np.array(input_data.matrix)

    
    if method == 'numpy':
        result = np.dot(M, X) + B

    else:

        result = [[0 for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):  
                for k in range(5):  
                    result[i][j] += M[i][k] * X[k][j]
                    
        # Add bias B
        for i in range(5):
            for j in range(5):
                result[i][j] += B[i][j]

    sigmoid_result = sigmoid(result)

    # Return the result as a JSON response
    return {"resultt": sigmoid_result.tolist()}



## EXAMPLE USAGES
##Create a 5 by 5 matrix, call the functions(matrix_multiplication)


if __name__ == "__main__":
    
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''
