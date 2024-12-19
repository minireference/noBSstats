# Exercise 11 str-prices-sum
prices = ["22.2", "10.1", "33.3"]
total = 0
for price in prices:
    total = total + float(price)
total

# ALT. using list comprehensions syntax
prices_float = [float(price) for price in prices]
sum(prices_float)