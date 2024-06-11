num_of_terms = int(input("How many terms? "))
fib_sequence = ""
a = 0
b = 1

for i in range(num_of_terms):
    fib_sequence += str(a) + " "
    a, b = b, a + b

print(f"Fibonacci sequence: {fib_sequence}")