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
# use the post decorator directly below this
'''
    Initialize M and B as np arrays
'''
def f(x):
    pass
 
#Implement the formula MX + B
#Have two function one using numpy and another not using numpy
#Return 

#initialize x as a 5 * 5 matrix

#Make a call to the function

#Recreate the function with the sigmoid Function

if __name__ == "__main__":
    uvicorn.run(app)

'''
    Create a requirements.txt
    Upload to render
'''

