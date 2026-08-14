import pandas as pd

print("Pandas is working.", "Yay!")
print(pd.__version__)

def add(x,y):
    return x + y

def mult(x,y):
    return x*y

def triangle(n):
    if n==1:
        return 1
    else:
        return add(n,triangle(n-1))
print(triangle(10))

print(add(2,5))

def factorial(n):
    if n==1:
        return 1 
    else:
        return mult(factorial(n-1),n)

print(factorial(10))


