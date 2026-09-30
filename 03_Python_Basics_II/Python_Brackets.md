# Python Brackets

The main idea:

```python
something(...)    # call something
object[...]       # select something
[...]             # create a list
{...}             # create a dictionary
```

## 1. Parentheses `( )`

Use them to **call a function or method**:

```python
print("Hello")
len(x)
x.copy()
np.mean(x)
```

Think: `something(...)` means **run/call something**.

They also group calculations:

```python
(2 + 3) * 4
```

And they can create a tuple:

```python
point = (10, 20)
```

## 2. Square brackets `[ ]`

Use them to **create a list**:

```python
years = [2022, 2023, 2024]
```

Or to **select something**:

```python
years[0]   # first element
years[1]   # second element
```

Python starts counting at `0`.

You can also select several elements:

```python
x[1:4]     # positions 1, 2, 3
x[:3]      # first three elements
x[-1]      # last element
x[::-1]    # reverse order
```

## 3. Curly brackets `{ }`

Usually used to create a dictionary:

```python
capitals = {
    "Denmark": "Copenhagen",
    "Sweden": "Stockholm"
}
```

To select from it, use square brackets:

```python
capitals["Denmark"]
```

So:

```python
{"Denmark": "Copenhagen"}   # create dictionary
capitals["Denmark"]         # select from dictionary
```

## 4. Why `[[ ... ]]`?

This usually means a list inside another list:

```python
x = [[1, 2], [3, 4]]
```

For example:

```python
A = np.array([[1, 2],
              [3, 4]])
```

Think of the inner lists as rows.

You can also select twice:

```python
x[0]       # [1, 2]
x[0][1]    # 2
```

## 5. NumPy rows and columns

```python
A[0, 1]    # row 0, column 1
A[0, :]    # row 0, all columns
A[:, 0]    # all rows, column 0
```

Here `:` means **all**.

## 6. Read complicated expressions from the inside out

```python
print(np.mean(x[1:4]))
```

Read:

```python
x[1:4]             # select
np.mean(x[1:4])    # calculate mean
print(...)         # print result
```

## The four patterns to remember

```python
function(...)      # CALL
object[...]        # SELECT
[...]              # CREATE A LIST
{label: value}     # CREATE A DICTIONARY
```

If a line looks confusing, read it **from the inside out**.
