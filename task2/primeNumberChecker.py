def isPrime(num):
    if num > 1:
        for i in range(2, num):
            if (num % i) == 0:
                return False
        else:
            return True
    else:
        return False

number = int(input("Enter a number: "))

if isPrime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")
