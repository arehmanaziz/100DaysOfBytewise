def longest_palindromic_substring(s):
    if not s:
        return ""


    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]


    longest = ""
    for i in range(len(s)):
        # Check for odd-length palindromes (single character center)
        odd_palindrome = expand_around_center(i, i)
        if len(odd_palindrome) > len(longest):
            longest = odd_palindrome

        # Check for even-length palindromes (pair of characters center)
        even_palindrome = expand_around_center(i, i + 1)
        if len(even_palindrome) > len(longest):
            longest = even_palindrome

    return longest

# Example usage
input_string1 = "babad"
output1 = longest_palindromic_substring(input_string1)
print(f"Longest palindromic substring in '{input_string1}' is '{output1}'")

input_string2 = "cbbd"
output2 = longest_palindromic_substring(input_string2)
print(f"Longest palindromic substring in '{input_string2}' is '{output2}'")
