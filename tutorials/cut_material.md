
### Useful anything-to-int conversions

# str --> int
number_as_str = '65'
print(number_as_str, 'has type', type(number_as_str))
number = int(number_as_str)
print(number, 'has type', type(number))

#  float --> int
precise_number_as_float = 3.2
print(precise_number_as_float, 'has type', type(precise_number_as_float))
rounded_number = int(precise_number_as_float)
print(rounded_number, 'has type', type(rounded_number))

#  bool --> int
true_value = True
false_value = False
print('When converted to an int, True is', int(True), 'and False is', int(False))


### Useful anything-to-float conversions

# str --> float
print(float('1.2'))  # note using decimal dot . not decimal , as in Europe
print(float('1e6'))  # one million = 1000000 = 1x10^6. The e is shorthand for x10^
print(float('4'))

# int/int --> float autoconversion
print('If you divide an integrer by an integer in Python using the / operator...')
print('...you get a float number', 6/5, 'has type', type(6/5))

print('There is also an divistion without autoconversion operator // which ...')
print('...you get an integer', 6//5, 'has type', type(6//5))



### Common anything-to-str conversions

# use the str function
str1 = str(42)
str2 = str(4.2)
str3 = str(True)
str4 = 'a string'
print('string representations', str1, str2, str3, str4)

# float -> str
import math
pi = math.pi  # constant equal to ratio of circumference to diameter of a circle
print('str(math.pi) =', str(math.pi))  # defaults to max precision

# control precision and print formatting using format. For more info, see
# https://python-reference.readthedocs.io/en/latest/docs/functions/format.html
print('pi to two decimals', '{:.2f}'.format(pi))
print('pi to seven decimals', '{:.7f}'.format(pi))




### Common anything-to-boolean conversions

Up until now we were in the land of math and text manipulation of numbers, which all make sense intuitively. 

Next we'll talk about the boolean values of Python expressions. This will feel a little weird at first since we're forcing some arbitray Python object (could be a number, a string, a list, a dictionary, etc) and asking the questions is the value `True` or `False` ?

There is a specific convention about which values are considered "truthy" (i.e. get converted to `True` when converted to boolean using `bool`) and which expressions are falsy (i.e. get converted to `False` when passed through `bool`). 

I know this seems like boring details, but please read the next example carefully because "truthyness" and "falsyness" will play a big role in coding: every time you use an if or elif statement in Python, we are implicitly calling `bool` on an expression so it's a good idea to get to know what `bool` does to different types of variables.


# int --> bool
print('Any non-zero integer is considered as True')
print(bool(1), bool(-2), bool(10000))
print('Zero is False')
print(bool(0), bool(-0), bool(int('0')))

print('\n')

# float --> bool
print('Any non-zero float is considered as True')
print(bool(1.0), bool(-2.0), bool(10000.0))
print('Zero is False')
print(bool(0.0), bool(-0.0), bool(float('0.0')))


# str --> bool
print('Any non-empty string is considered True')
print(bool('as'), bool(''))

print('\n')

# list --> bool
print('Any non-empty list is considered True')
print(bool([1]), bool([1,2]), bool(range(0,10000)))
print('Empty list is considered False')
print(bool([]), bool(list()), bool(list('[]')))

print('\n')

# dict --> bool
print('Any non-empty dict is considered True')
print(bool({1:11}), bool({1:11,-2:22}))
print('Empty dict is considered False')
print(bool({}), bool(dict()))








### Bonus topics
  - Command line basics: similar to how in computing user interfaces click (or  double click) means "run <program>", a command line interface allows us to run programs by simply typing out their name, as in    program    and pressing. Enter. This is where the notion of "calling a program/function" comes from, you   just write it's name in a command line prompt and this is equivalent to "calling it" (making it run, just click clicking on it)
  - Command line scripts: run any python code on command line using  python
  myscript.py (or if configured as an executable script, simply ./myscript.py)
  - Documentation: READMEs, __doc__ strings, and other technical altruism basics
  - Debugging (intro to JupyterLab debugger)
  - Code Testing (e.g. test_fun: a function that checks that `fun` returns the expected outputs on some set of inputs)
  - Algorithms: different approaches for solving computational problem

  

This notebook contains some exercises to get you familiar with basic Python programming notions.




There are multiple data types in Python 
Similar to the notion of number sets in math $\mathbb{Z}$, $\mathbb{R}$ etc., computers variables also come in different types



Just enough to be dangerous with numbers and for loops.
Nothing fancy. We'll be mostly using Python as a basic calculator for math expressions,


and "boring" calculations that require repeating the same action many times (for loops).

So if you know how to use a calculator, then you know how to use Python too.


If you remember some basic math concepts like variables, expressions, and functions,
then you already know most of all there is to know!



We can access individual values within the series using the square brackets, and zero-based indexing. Recall this is the same syntax that we use to access elements in a Python list.
>>> s[0]

If you want just the values, get `.values` first, then slice

    >>> s.values[0:3]
    array([3, 5, 7])

or slice then get the values

    >>> s[0:3].values
    array([3, 5, 7])