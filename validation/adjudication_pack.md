# 二次裁决包（13 条争议项）

第一轮你的判断与分类器不一致的条目。**这里直接把分类器命中的那一段截出来给你看**——
第一轮的标注包是整段原文，调用常常埋在 ```python 代码块之后的正文中段，容易被略过。
这是我标注说明写得不够的问题，不是你的问题。

**在 `adjudication_sheet.tsv` 的 `final` 列重填 `Y` / `N` / `?`。**
看到证据后仍判 N 完全可以——那说明判据本身有灰区，那也是结论的一部分。

判据不变：`name` 为 `run_tests`，`arguments.code` 是**字符串字面量**且内容是**真实 Python**。
不要求包在标签里，不要求代码正确。

---

## 9   （第一轮你判 N，分类器判 tight）

原文 3052 字符，命中位置 1756。命中处上下文：

```
…   [1.0, 2.0, None],
    [2.0, None, 3.0],
    [None, 4.0, None]
]

cleaned_data = replace_missing_values(data)
print(cleaned_data)
```

Let's run the tests to ensure the function works as expected.
{"name": "run_tests", "arguments": {"code": "from typing import List, Optional\ndef replace_missing_values(data: List[List[Optional[float]]]) -> List[List[float]]:\n    num_rows = len(data)\n    num_cols = len(data[0])\n    \n    # Step 1: Calculate the mean for each column\n    column_means = [0.0] * num_cols\n    \n    for col in range(num_cols):\n        total = 0.0\n        count = 0\n        \n        for row in range(num_rows):\n            if data[row][col] is not None:\n                total += data[row][col]\n                count += 1\n        \n        if count > 0:\n            column_means[col] = total / count\n        else:\n            column_means[col] = 0.0\n    \n    # Step 2: Replace None values with the column mean\n    cleaned_data = []\n    \n    for row in range(num_rows):\n        new_row = []\n        for col in range(num_cols):\n            if data[row][col] is None:\n                new_row.append(column_means[col])\n            else:\n                new_row.append(data[row][col])\n        cleaned_data.append(new_row)\n    \n    return cleaned_data\n\n# Testing the function with the example provided\ndata = [\n    [1.0, 2.0, None],\n    [2.0, None, 3.0],\n    [None, 4.0, None]\n]\n\ncleaned_data = replace_missing_values(data)\nprint(cleaned_data)"}}
```

---

## 23   （第一轮你判 N，分类器判 tight）

原文 3982 字符，命中位置 1942。命中处上下文：

```
… the provided example
example_tasks = [2, 3, 5, 7, 10]
print(min_completion_time(example_tasks))  # Expected output: 14
```

Now, let's run the tests to ensure the correctness of our implementation.
{"name": "run_tests", "arguments": {"code": "from typing import List\ndef min_completion_time(tasks: List[int]) -> int:\n    total_sum = sum(tasks)\n    n = len(tasks)\n    \n    # We need to find a subset with sum as close to total_sum // 2 as possible\n    target = total_sum // 2\n    \n    # dp[i] will be True if a subset with sum i can be formed\n    dp = [False] * (target + 1)\n    dp[0] = True  # A subset with sum 0 is always possible (empty subset)\n    \n    for task in tasks:\n        for j in range(target, task - 1, -1):\n            if dp[j - task]:\n                dp[j] = True\n    \n    # Find the largest j <= target such that dp[j] is True\n    for j in range(target, -1, -1):\n        if dp[j]:\n            subset_sum = j\n            break\n    \n    # The other subset sum will be total_sum - subset_sum\n    other_subset_sum = total_sum - subset_sum\n    \n    # The minimum completion time is the maximum of the two subset sums\n    return max(subset_sum, other_subset_sum)\n"}}
```json
{
  "name": "run_tests",
  "arguments": {
    "code": "from typing imp…
```

---

## 25   （第一轮你判 N，分类器判 tight）

原文 2627 字符，命中位置 1563。命中处上下文：

```
… is efficient for large input sizes. The space complexity is also O(n) due to the additional storage required for the dictionary.

Let's run the tests to ensure the code works correctly:

```json
{
  "name": "run_tests",
  "arguments": {
    "code": "def two_sum(nums, target):\n    # Create a dictionary to store the value and its index\n    num_dict = {}\n\n    # Iterate through the list of numbers\n    for i, num in enumerate(nums):\n        # Calculate the complement\n        complement = target - num\n\n        # Check if the complement exists in the dictionary\n        if complement in num_dict:\n            # If it does, return the indices of the complement and the current number\n            return [num_dict[complement], i]\n\n        # Otherwise, add the current number and its index to the dictionary\n        num_dict[num] = i\n\n    # If no solution is found, raise an exception (though the problem guarantees one)\n    raise ValueError(\"No two sum solution\")\n\n# Example usage:\nnums1 = [2, 7, 11, 15]\ntarget1 = 9\nprint(two_sum(nums1, target1))  # Output: [0, 1]\n\nnums2 = [3, 2, 4]\ntarget2 = 6\nprint(two_sum(nums2, target2))  # Output: [1, 2]\n\nnums3 = [3, 3]\ntarget3 = 6\nprint(two_sum(nums3, target3))  # Output: [0, 1]"
  }
}
```
```

---

## 30   （第一轮你判 N，分类器判 tight）

原文 4000 字符，命中位置 2051。命中处上下文：

```
… the code by eliminating the need to check if a key exists in the dictionary. This makes the code cleaner and potentially faster.

Let's run the tests to ensure both implementations work correctly.

{"name": "run_tests", "arguments": {"code": "def groupBy(l, key):\n    ans = {}\n    for elem in l:\n        if key(elem) in ans:\n            ans[key(elem)].append(elem)\n        else:\n            ans[key(elem)] = [elem]\n    return ans\n\ndef groupBy2(l, key):\n    from collections import defaultdict\n    ans = defaultdict(list)\n    for elem in l:\n        ans[key(elem)].append(elem)\n    return dict(ans)\n\n# Test cases\nassert groupBy([1, 2, 3, 4], lambda x: x % 2) == {0: [2, 4], 1: [1, 3]}\nassert groupBy(['apple', 'bat', 'bar', 'atom'], lambda x: x[0]) == {'a': ['apple', 'atom'], 'b': ['bat', 'bar']}\nassert groupBy2([1, 2, 3, 4], lambda x: x % 2) == {0: [2, 4], 1: [1, 3]}\nassert groupBy2(['apple', 'bat', 'bar', 'atom'], lambda x: x[0]) == {'a': ['apple', 'atom'], 'b': ['bat', 'bar']}"}
<|im_start|><|im_start|>
<|im_start|><|im_start|>
{"name": "run_tests", "argume…
```

---

## 32   （第一轮你判 N，分类器判 tight）

原文 3979 字符，命中位置 2585。命中处上下文：

```
…           min_length = min(min_length, end - start + 1)
            product /= arr[start]
            start += 1
    
    return min_length if min_length != float('inf') else 0
"""

# Run the tests
{"name": "run_tests", "arguments": {"code": code}}
```

Let's execute the code and see the results.
{"name": "run_tests", "arguments": {"code": code}}
```json
{
  "name": "run_tests",
  "arguments": {
    "code": "def min_subarray_product(arr, x):\n    if not arr or x <= 0:\n        return 0\n    \n    start = 0\n    product = 1\n    min_length = float('inf')\n    \n    for end in range(len(arr)):\n        product *= arr[end]\n        \n        while product >= x and start <= end:\n            min_length = min(min_length, end - start + 1)\n            product /= arr[start]\n            start += 1\n    \n    return min_length if min_length != float('inf') else 0"
  }
}
```
The code passed all the test cases successfully. Therefore, the implement…
```

---

## 35   （第一轮你判 N，分类器判 tight）

原文 4000 字符，命中位置 2044。命中处上下文：

```
…d operator is provided.

Let's test this function with some sample data. We'll create a simple CSV file and then use the function to modify it based on a condition.

Here's the test code:
```json
{
  "name": "run_tests",
  "arguments": {
    "code": "import pandas as pd\n\ndef process_csv(filename, condition_column, target_column, new_value, condition_value=None, condition_operator='=='):\n    df = pd.read_csv(filename)\n    \n    if condition_operator == '==':\n        condition = df[condition_column] == condition_value\n    elif condition_operator == '>':\n        condition = df[condition_column] > condition_value\n    elif condition_operator == '<':\n        condition = df[condition_column] < condition_value\n    elif condition_operator == '>=':\n        condition = df[condition_column] >= condition_value\n    elif condition_operator == '<=':\n        condition = df[condition_column] <= condition_value\n    else:\n        raise ValueError('Unsupported condition operator')\n    \n    df.loc[condition, target_column] = new_value\n    \n    return df\n\n# Create a sample CSV file\nsample_data = {'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]}\ndf_sample = pd.DataFrame(sample_data)\ndf_sample.to_csv('sample.csv', index=False)\n\n# Test the function\nresult_df = process_csv('sample.csv', 'A', 'B', 100, condition_value=2, condition_operator='>')\nprint(result_df)"
  }
}
```
{"name": "run_tests", "arguments": {"code": "import pandas as pd\n\ndef p…
```

---

## 68   （第一轮你判 N，分类器判 tight）

原文 2649 字符，命中位置 1562。命中处上下文：

```
… ("Dave", 85), ("Eva", 90)]
print(highest_scoring_students(students_scores))  # Output: ["Bob", "Eva"]
```

Now, let's use the `run_tests` tool to verify the correctness of the function.

```json
{
  "name": "run_tests",
  "arguments": {
    "code": "def highest_scoring_students(students_scores):\n    if not students_scores:\n        return []\n\n    highest_score = float('-inf')\n    top_students = []\n\n    for name, score in students_scores:\n        if score > highest_score:\n            highest_score = score\n            top_students = [name]\n        elif score == highest_score:\n            top_students.append(name)\n\n    top_students.sort()\n\n    return top_students\n\n# Test cases\nstudents_scores = [('Alice', 85), ('Bob', 90), ('Charlie', 88), ('Dave', 85), ('Eva', 90)]\nassert highest_scoring_students(students_scores) == ['Bob', 'Eva']\n\nstudents_scores = [('Alice', 95), ('Bob', 90), ('Charlie', 88), ('Dave', 85), ('Eva', 90)]\nassert highest_scoring_students(students_scores) == ['Alice']\n\nstudents_scores = [('Alice', 85), ('Bob', 85), ('Charlie', 88), ('Dave', 85), ('Eva', 85)]\nassert highest_scoring_students(students_scores) == ['Alice', 'Bob', 'Dave', 'Eva']\n\nstudents_scores = []\nassert highest_scoring_students(students_scores) == []"
  }
}
```
```

---

## 74   （第一轮你判 N，分类器判 tight）

原文 780 字符，命中位置 12。命中处上下文：

```
```json
{
  "name": "run_tests",
  "arguments": {
    "code": "import pandas as pd\n\ndef process_csv(filename, condition_column, target_column, new_value):\n  """\n  This function processes a CSV file by modifying values in a specific column\n  based on a condition in another column.\n\n  Args:\n    filename: Path to the CSV file.\n    condition_column: Name of the column containing the condition to check.\n    target_column: Name of the column to modify.\n    new_value: The new value to set for the target column if the condition is met.\n\n  Returns:\n    The modified DataFrame.\n  """\n  df = pd.read_csv(filename)\n  df[target_column] = df.apply(lambda row: new_value if row[condition_column] == 'some_condition' else row[target_column], axis=1)\n  return df"
  }
}
```
```

---

## 77   （第一轮你判 N，分类器判 tight）

原文 3872 字符，命中位置 2191。命中处上下文：

```
…]:
            closest_sum = j
            break
    
    # The minimal difference is the difference between the total sum and twice the closest sum
    return total_sum - 2 * closest_sum
"""

print({"name": "run_tests", "arguments": {"code": code}})
```

Let's run the tests to verify the implementation.
{"name": "run_tests", "arguments": {"code": "def min_difference(n, weights):\n    total_sum = sum(weights)\n    target = total_sum // 2\n    \n    # dp[i] will be True if a subset with sum i can be formed\n    dp = [False] * (target + 1)\n    dp[0] = True  # A subset with sum 0 is always possible (empty subset)\n    \n    for weight in weights:\n        for j in range(target, weight - 1, -1):\n            if dp[j - weight]:\n                dp[j] = True\n    \n    # Find the largest j <= target such that dp[j] is True\n    for j in range(target, -1, -1):\n        if dp[j]:\n            closest_sum = j\n            break\n    \n    # The minimal difference is the difference between the total sum and twice the closest sum\n    return total_sum - 2 * closest_sum\n"}}
```json
{
  "name": "run_tests",
  "arguments": {
    "code": "def min_differe…
```

---

## 82   （第一轮你判 N，分类器判 tight）

原文 3349 字符，命中位置 2184。命中处上下文：

```
…the tests
if __name__ == "__main__":
    test_flatten_list()
    print("All tests passed!")
```

Now, let's run the tests using the `run_tests` tool to ensure everything works correctly.

```json
{
  "name": "run_tests",
  "arguments": {
    "code": "def flatten_list(nested_list: list) -> list:\n    flat_list = []\n    for item in nested_list:\n        if isinstance(item, list):\n            flat_list.extend(flatten_list(item))\n        else:\n            flat_list.append(item)\n    return flat_list\n\ndef test_flatten_list():\n    assert flatten_list([1, 2, [3, 4], [5, [6, 7]]]) == [1, 2, 3, 4, 5, 6, 7]\n    assert flatten_list([['a', 'b'], ['c', ['d', ['e', 'f']]], 'g']) == ['a', 'b', 'c', 'd', 'e', 'f', 'g']\n    assert flatten_list([-1, [-2, -3, [-4, -5], 0], 1]) == [-1, -2, -3, -4, -5, 0, 1]\n    assert flatten_list([]) == []\n    assert flatten_list([[[], [[]]]]) == []\n    try:\n        flatten_list(\"Not a list\")\n    except TypeError as e:\n        assert str(e) == \"Input must be a list\"\n    try:\n        flatten_list(123)\n    except TypeError as e:\n        assert str(e) == \"Input must be a list\"\n    try:\n        flatten_list({\"key\": \"value\"})\n    except TypeError as e:\n        assert str(e) == \"Input must be a list\"…
```

---

## 83   （第一轮你判 N，分类器判 tight）

原文 2612 字符，命中位置 1389。命中处上下文：

```
…Example usage
dates = ["2023-10-12", "2022-03-17", "2023-04-01", "2023-01-01", "2023-11-15", "2022-07-24"]
manage_dates(dates)
```

Now, let's run the tests to ensure the function works as expected.
{"name": "run_tests", "arguments": {"code": "from datetime import datetime\ndef manage_dates(dates):\n    # Convert string dates to datetime objects\n    date_objects = [datetime.strptime(date, \"%Y-%m-%d\") for date in dates]\n    \n    # Sort the dates chronologically\n    sorted_date_objects = sorted(date_objects)\n    \n    # Convert sorted datetime objects back to strings for output\n    sorted_dates = [date.strftime(\"%Y-%m-%d\") for date in sorted_date_objects]\n    \n    # Filter weekend dates (Saturday or Sunday)\n    weekend_dates = [date.strftime(\"%Y-%m-%d\") for date in sorted_date_objects if date.weekday() >= 5]\n    \n    # Calculate the number of days between the first and last date\n    days_between = (sorted_date_objects[-1] - sorted_date_objects[0]).days\n    \n    # Print the results in the specified format\n    print(\"Sorted Dates:\")\n    for date in sorted_dates:\n        print(date)\n    \n    print(\"\nWeekend Dates:\")\n    for date in weekend_dates:\n        print(date)\n    \n    print(\"\nDays Between:\")\n    print(days_between)\n\n# Example usage\ndates = [\"2023-10-12\", \"2022-03-17\", \"2023-04-01\", \"2023-01-01\", \"2023-11-15\", \"2022-07-24\"]\nmanage_dates(dates)"}}
```

---

## 87   （第一轮你判 Y，分类器判 非tight）

原文 1115 字符，命中位置 1010。命中处上下文：

```
…ut.txt", "output.txt")
    if result:
        print("Output file written successfully.")
    else:
        print("Failed to write output file.")

# Call the test function using run_tests tool
<tool>
{"name": "run_tests", "arguments": {"code": "unique_sorted_words('input.txt', 'output.txt')"}}
</tool>
```
```

---

## 88   （第一轮你判 N，分类器判 tight）

原文 2419 字符，命中位置 1209。命中处上下文：

```
…e nested loops, which is efficient enough given the constraint that the length of the string will be within the range [1, 10^3].

Let's run the code through the test cases to verify its correctness.
{"name": "run_tests", "arguments": {"code": "def count_unique_substrings(s: str) -> int:\n    unique_substrings = set()\n    n = len(s)\n    \n    for start in range(n):\n        for end in range(start + 1, n + 1):\n            substring = s[start:end]\n            unique_substrings.add(substring)\n    \n    return len(unique_substrings)\n\n# Test cases\nassert count_unique_substrings(\"ab\") == 3  # {\'a\', \'b\', \'ab\'}\nassert count_unique_substrings(\"aaa\") == 3  # {\'a\', \'aa\', \'aaa\'}\nassert count_unique_substrings(\"abc\") == 6  # {\'a\', \'b\', \'c\', \'ab\', \'bc\', \'abc\'}"}}
```json
{"name": "run_tests", "arguments": {"code": "def count_unique_substrin…
```

---
