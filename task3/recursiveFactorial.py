def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

n = 5
print(f"The factorial of {n} is {factorial(n)}")

