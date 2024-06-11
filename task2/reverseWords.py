string = input("Eneter a string: ")
new_string = " ".join(string.split(" ")[::-1])
print(new_string)