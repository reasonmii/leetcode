# Concept

## Approval Drop

Capital approval rates have gone down for our overall approval rate. </br>
Let’s say last week it was 85% and the approval rate went down to 82% this week which is a statistically significant reduction. </br>
The first analysis shows that all approval rates stayed flat or increased over time when looking at the individual products: </br>
Product 1: 84% to 85% week over week </br>
Product 2: 77% to 77% week over week </br>
Product 3: 81% to 82% week over week </br>
Product 4: 88% to 88% week over week </br>
What could be the cause of the decrease? </br>

This is becaue the weight in overall approval of individual product has changes. This is **Simpson paradox** as mentioned by other users.
- Simpson’s Paradox occurs when a trend shows in several groups but either disappears or is reversed when combining the data.

## Same Algorithm Different Success
Why would the same machine learning algorithm generate different success rates using the same dataset? </br>
Note: When they ask us an ambiguous question, we need to gather context and restate it in a way that’s clear for us to answer. </br>

# Statistics

## Same Side Probability

Suppose we have two coins. One is fair and the other biased where the probability of it coming up heads is 3⁄4. </br>
Let’s say we select a coin at random and flip it two times. What is the probability that both flips result in the same side?

- Choose 1 : 1/2
- Heads : 1/2 * (1/2 * 1/2) + 1/2 * (3/4 * 3/4) = 13/32
- Tails : 1/2 * (1/2 * 1/2) + 1/2 * (1/4 * 1/4) = 5/32
- tot : 18/32 = 9/16 = 0.5625

## First to Six
Amy and Brad take turns in rolling a fair six-sided die. Whoever rolls a “6” first wins the game. Amy starts by rolling first. What’s the probability that Amy wins?

P(Amy|first roll) = 1/6 </br>
P(Brad|first roll) = 5/6 * 1/6 </br>
P(Amy|second roll) = 5/6 * 5/6 * 1/6 </br>
P(Brad|second roll) = 5/6 * 5/6 * 5/6 * 1/6 </br>

Geometric series
- 1/6 + (5/6)^2 * 1/6 + (5/6)^4 * 1/6 + (5/6)^6 * 1/6 + ...
- $S = a + ar + ar^2 + ar^3 + ...$
  - $Sr = ar + ar^2 + ar^3 + ar^4 + ...$
  - $Sr = S - a$
  - $S(1-r) = a$
  - $S = a / (1-r)$

Infinite Geometric series : 1/6 / (1 - 25/36) = 6/11 </br>
Answer : 6/11

## Raining in Seattle

You are about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it’s raining. Each of your friends has a 2⁄3 chance of telling you the truth and a 1⁄3 chance of messing with you by lying. All 3 friends tell you that “Yes” it is raining. </br>

What is the probability that it’s actually raining in Seattle? </br>

Assume : P(rain) = 0.5 </br>
P(all truth) = 2/3 * 2/3 * 2/3 = 8/27 </br>
P(all lie) = 1/3 * 1/3 * 1/3 = 1/27 </br>
P(yes) = P(yes|rain) * P(rain) + P(yes|not rain) * P(not rain) </br>
conditional probability = 8/27 / (8/27 + 1/27) = 8/27 / (1/3) = 8/9

# SQL

## Random SQL Sample

```SELECT * FROM big_table ORDER BY RAND() limit 1```

## HR Salary Reporting

```
select job_title
     , sum(salary) total_salaries
     , sum(overtime_hours * overtime_rate) total_overtime_payments
     , sum(salary) + sum(overtime_hours * overtime_rate) total_compensation
from employees
group by 1
```

# Python

## Merge Sorted Lists

```
def merge_list(list1, list2):

    i = len(list1) - 1 # 2
    j = len(list2) - 1 # 2
    rst = []

    while list1 and list2:
        if list1[i] >= list2[j]:
            rst = [list1.pop()] + rst
            i -= 1
        else:
            rst = [list2.pop()] + rst
            j -= 1

    return list1 + list2 + rst
```
