str1 = input("Eneter first string: ")
str2 = input("Eneter second string: ")

if sorted(str1) == sorted(str2):
    print(f"{str1} and {str2} are anagrams")
else:
    print(f"{str1} and {str2} are not anagrams")