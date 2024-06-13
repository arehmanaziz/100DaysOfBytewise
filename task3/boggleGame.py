dictionary = ["DATA", "HALO", "HALT", "QUIZ", "GO", "WENT", "SAG", "BEAT", "TOTAL", "DOG", "FOR"]
n = len(dictionary)
M = 4
N = 4

def isWord(Str):
    return Str in dictionary

def findWordsUtil(boggle, visited, i, j, Str, found_words):
    visited[i][j] = True
    Str = Str + boggle[i][j]

    if isWord(Str):
        found_words.add(Str)
    
    row = i - 1
    while row <= i + 1 and row < M:
        col = j - 1
        while col <= j + 1 and col < N:
            if row >= 0 and col >= 0 and not visited[row][col]:
                findWordsUtil(boggle, visited, row, col, Str, found_words)
            col += 1
        row += 1

    visited[i][j] = False

def findWords(boggle):
    visited = [[False for _ in range(N)] for _ in range(M)]
    found_words = set()
    
    for i in range(M):
        for j in range(N):
            findWordsUtil(boggle, visited, i, j, "", found_words)
    
    return found_words

boggle = [["D", "A", "T", "H"], ["C", "G", "O", "A"], ["S", "A", "T", "L"], ["B", "E", "D", "G"]]

print("Following words of the dictionary are present")
found_words = findWords(boggle)
print(found_words)
