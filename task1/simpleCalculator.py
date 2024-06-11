num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
operation = input("Enter operation (+, -, *, /): ")

if operation == "+":
    print(f"The result is {num1 + num2}")
elif operation == "-":
    print(f"The result is {num1 - num2}")
elif operation == "*":
    print(f"The result is {num1 * num2}")
else:
    print(f"The result is {num1 / num2}")