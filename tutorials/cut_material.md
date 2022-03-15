
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