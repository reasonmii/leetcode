
### Merge Sorted Lists

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

### First to Six
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

### Raining in Seattle

You are about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it’s raining. Each of your friends has a 2⁄3 chance of telling you the truth and a 1⁄3 chance of messing with you by lying. All 3 friends tell you that “Yes” it is raining. </br>

What is the probability that it’s actually raining in Seattle? </br>

Assume : P(rain) = 0.5 </br>
P(all truth) = 2/3 * 2/3 * 2/3 = 8/27 </br>
P(all lie) = 1/3 * 1/3 * 1/3 = 1/27 </br>
P(yes) = P(yes|rain) * P(rain) + P(yes|not rain) * P(not rain) </br>
conditional probability = 8/27 / (8/27 + 1/27) = 8/27 / (1/3) = 8/9


















