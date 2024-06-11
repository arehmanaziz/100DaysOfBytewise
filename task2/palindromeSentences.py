string = input("Enter a sentence: ")

nstring = string.lower()
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
new_string = ""
for char in nstring:
    if char not in punctuations:
        new_string += char

new_string = "".join(new_string.split(" "))

if new_string == new_string[::-1]:
    print(f"{string} is a palindrome.")
else:
    print(f"{string} is not a palindrome.")