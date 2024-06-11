string = input("Enter a string: ")
vowels = "aeiou"
count = sum(1 for char in string if char in vowels)
print(f"The number of vowels is: {count}")