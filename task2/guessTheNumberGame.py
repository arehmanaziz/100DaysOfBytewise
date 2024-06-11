import random

random_number = random.randint(1, 100)
while True:
    num = int(input("Guess the number (1 to 100): "))
    if num == random_number:
        print("You guessed it!")
        break
    elif num > random_number:
        print("Too high")
    else:
        print("Too low")