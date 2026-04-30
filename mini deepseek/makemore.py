import json
import random

data = []

# Addition - with more steps
for _ in range(15):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    total = a + b
    
    data.append({
        "input": f"Solve: {a} + {b}",
        "output": f"""Step 1: Break down the numbers
Step 2: Add the hundreds: {a//100*100} + {b//100*100} = {(a//100 + b//100)*100}
Step 3: Add the tens and units: {a%100} + {b%100} = {a%100 + b%100}
Step 4: Combine both parts: {(a//100 + b//100)*100} + {a%100 + b%100} = {total}
Step 5: Verify the addition
Answer: {total}"""
    })

# Subtraction - with more steps
for _ in range(15):
    a = random.randint(300, 999)
    b = random.randint(100, a-100)
    diff = a - b
    
    data.append({
        "input": f"Solve: {a} - {b}",
        "output": f"""Step 1: Align the numbers by place value
Step 2: Subtract the units: {a%10} - {b%10}
Step 3: Subtract the tens: {(a//10)%10} - {(b//10)%10}
Step 4: Subtract the hundreds: {a//100} - {b//100}
Step 5: Combine the results: {diff}
Step 6: Double-check by adding back: {b} + {diff} = {a} ✔
Answer: {diff}"""
    })

# Multiplication - more detailed steps
for _ in range(15):
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    product = a * b
    
    data.append({
        "input": f"Solve: {a} × {b}",
        "output": f"""Step 1: Break {b} into tens and units: {b//10*10} + {b%10}
Step 2: Multiply {a} by units digit ({b%10}): {a} × {b%10} = {a*(b%10)}
Step 3: Multiply {a} by tens digit ({b//10}) and shift left: {a} × {b//10} = {a*(b//10)}, then ×10 = {a*(b//10)*10}
Step 4: Add both partial products: {a*(b%10)} + {a*(b//10)*10} = {product}
Step 5: Final verification
Answer: {product}"""
    })

# Division - more steps
for _ in range(15):
    b = random.randint(2, 20)
    multiplier = random.randint(10, 60)
    a = b * multiplier
    quotient = a // b
    
    data.append({
        "input": f"Solve: {a} ÷ {b}",
        "output": f"""Step 1: Understand that we need to find how many times {b} fits into {a}
Step 2: Estimate: {a} ÷ {b} is roughly between {multiplier-5} and {multiplier+5}
Step 3: Perform the division: {b} × {quotient} = {a}
Step 4: Check for remainder: {a} - ({b} × {quotient}) = 0
Step 5: Conclusion - exact division
Answer: {quotient}"""
    })

# Modulo
for _ in range(10):
    a = random.randint(50, 300)
    b = random.randint(2, 15)
    remainder = a % b
    
    data.append({
        "input": f"Solve: {a} mod {b}",
        "output": f"""Step 1: Division: {a} ÷ {b} = {a//b} with some remainder
Step 2: Multiply divisor by quotient: {b} × {a//b} = {b*(a//b)}
Step 3: Subtract from original number: {a} - {b*(a//b)} = {remainder}
Step 4: The remainder is always less than the divisor ({remainder} < {b})
Answer: {remainder}"""
    })

# Square roots (perfect squares)
for _ in range(10):
    n = random.randint(4, 40)
    sq = n * n
    data.append({
        "input": f"Solve: √{sq}",
        "output": f"""Step 1: Recognize that we are looking for a number that when multiplied by itself gives {sq}
Step 2: Test nearby integers
Step 3: {n-1}² = {(n-1)**2}, which is too small
Step 4: {n}² = {sq}, which matches exactly
Step 5: Therefore, the square root is {n}
Answer: {n}"""
    })

# Powers
for _ in range(10):
    base = random.randint(2, 12)
    exp = random.randint(2, 5)
    result = base ** exp
    data.append({
        "input": f"Solve: {base}^{exp}",
        "output": f"""Step 1: Understand exponentiation as repeated multiplication
Step 2: {base}^1 = {base}
Step 3: {base}^2 = {base} × {base} = {base*base}
Step 4: Continue multiplying: {base}^{exp} = {base} × {base} × ... ({exp} times)
Step 5: Final calculation: {result}
Answer: {result}"""
    })

# Percentages - more steps
for _ in range(10):
    pct = random.choice([5, 10, 15, 20, 25, 30, 50])
    n = random.randint(50, 800)
    result = round((n * pct) / 100)
    
    data.append({
        "input": f"Solve: What is {pct}% of {n}?",
        "output": f"""Step 1: Convert percentage to decimal: {pct}% = {pct}/100 = {pct/100}
Step 2: Multiply: {n} × {pct/100}
Step 3: Alternative method: ({n} × {pct}) ÷ 100
Step 4: Calculation: ({n} × {pct}) = {n*pct}
Step 5: Divide by 100: {n*pct} ÷ 100 = {result}
Answer: {result}"""
    })

random.shuffle(data)

with open("examples.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} examples with more detailed steps")