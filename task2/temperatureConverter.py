def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def celsius_to_kelvin(celsius):
    return celsius + 273.15

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def fahrenheit_to_kelvin(fahrenheit):
    return (fahrenheit - 32) * 5/9 + 273.15

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32

def temperature_converter(temp, unit):

    if unit == 'C':
        fahrenheit = celsius_to_fahrenheit(temp)
        kelvin = celsius_to_kelvin(temp)
        print(f"{temp} degrees Celsius is equal to {fahrenheit:.2f} degrees Fahrenheit and {kelvin:.2f} degrees Kelvin.")
    
    elif unit == 'F':
        celsius = fahrenheit_to_celsius(temp)
        kelvin = fahrenheit_to_kelvin(temp)
        print(f"{temp} degrees Fahrenheit is equal to {celsius:.2f} degrees Celsius and {kelvin:.2f} degrees Kelvin.")
    
    else:
        celsius = kelvin_to_celsius(temp)
        fahrenheit = kelvin_to_fahrenheit(temp)
        print(f"{temp} degrees Kelvin is equal to {celsius:.2f} degrees Celsius and {fahrenheit:.2f} degrees Fahrenheit.")
    

temp = float(input("Enter the temperature: "))
unit = input("Enter the unit (C for Celsius, F for Fahrenheit, K for Kelvin): ").upper()

temperature_converter(temp, unit)
