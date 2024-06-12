n = int(input("Number of term: "))
a = 0
b = 1

for i in range(n):
    a, b = b, a + b

print(f"The {n}th Fibonacci number is {a}")