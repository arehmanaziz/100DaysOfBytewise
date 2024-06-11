num_of_terms = int(input("How many terms? "))
a = 0
b = 1

seq = []
for i in range(num_of_terms):
    seq.append(a)
    a, b = b, a + b

print(", ".join(map(str, seq)))