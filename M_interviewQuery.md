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

- Random Initialization
  - Many machine learning algorithms, especially those involving neural networks, use random initialization of weights and biases.
  - This randomness can lead to different outcomes each time the model is trained.
- Stochastic Nature of the Algorithm
  - Algorithms like stochastic gradient descent (SGD) introduce randomness through their iterative process, as they sample batches of data randomly in each epoch.
  - This can lead to different models and, consequently, different success rates.
- Data Splitting
  - If the dataset is split into training and testing sets randomly each time, different splits can result in different training and testing datasets.
  - This can cause variations in model performance due to the different compositions of these datasets.
- Cross-Validation
  - When using cross-validation, the data is split into different subsets multiple times.
  - Each split can lead to slightly different model training processes and, therefore, different success rates.
- Hyperparameter Tuning
  - If the hyperparameters of the algorithm are not fixed and are being tuned through methods like grid search or random search,
  - different hyperparameter values can lead to different model performances.
= Hardware and Computational Differences
  - Variations in the hardware or computational environment can lead to slight differences in numerical computations, which can affect the outcome.
  - ex) GPU vs. CPU, different versions of libraries, etc.
- Overfitting and Underfitting
  - If the model is overfitting or underfitting, small changes in the data or initialization can lead to significant changes in performance.
  - Overfitting to noise in the training data can especially cause high variance in the success rates.
- Data Preprocessing
  - Differences in how data is preprocessed can lead to different results.
  - Even small changes in preprocessing can impact the performance of the model.
  - ex) scaling, normalization, handling missing values

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

## Skewed Pricing

Let’s say that we’re building a model to predict real estate home prices in a particular city. We analyze the distribution of the home prices and see that the **homes values are skewed to the right.** Do we need to do anything or take it into consideration? If so, what should we do? </br>
Bonus: Let’s say you see your target distribution is **heavily left** instead. What do you do now?

**Right-Skewed** Distribution (Positively Skewed)
- Log Transform : $y = log(y)$
- Box-Cox Transform : more flexible transformation that includes a parameter lambda that can be adjusted to **best normalize the data**
  - $y = (y^{\lambda} - 1) / \lambda$
- Square Root Transform : $y = \sqrt{y}$

**Left-Skewed** Distribution (Negatively Skewed)
- Log Transform : $y = max(y) + 1 - y -> y = log(y)$
- Box Cox Transform : make $\lambda < 0$
- Square Root Transform : $y = max(y) + 1 - y -> y = \sqrt{y}$

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

## Employee Salaries (ETL Error)

```
select first_name
     , last_name
     , salary
from (
    select first_name
        , last_name
        , salary
        , rank()over(partition by first_name, last_name order by id desc) rk
    from employees
) as t
where rk = 1
```

## Download Facts

```
select d.download_date
     , a.paying_customer
     , ROUND(avg(downloads),2) average_downloads
from accounts a
inner join downloads d on a.account_id = d.account_id
group by 1, 2
```

## Sequentially Fill in Integers

```
WITH RECURSIVE t_rp AS (
    SELECT int_numbers
         , 1 as lvl
    FROM tbl_numbers
    UNION ALL
    SELECT int_numbers
         , lvl+1
    FROM t_rp
    where lvl+1 <= int_numbers
)
select int_numbers as seq_numbers
from t_rp
order by int_numbers
;
```

## Lowest Paid

```
with t as (
    select e.id employee_id
        , e.salary
        , count(*) completed_projects
    from employees e
    left join employee_projects ep on e.id = ep.employee_id
    left join projects p on ep.project_id = p.id
    where p.end_date is not null
    group by e.id
    having count(*) >= 2
)
select t.employee_id
     , t.salary
     , t.completed_projects
from employees e
inner join t on e.id = t.employee_id
order by e.salary
limit 3
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

## Find the Missing Number

```
def missing_number(nums):

  nums = sorted(nums)  
  for i in range(1, len(nums)):
    if nums[i-1] != nums[i] - 1:
      return nums[i] - 1

  return 0
```

## Target Value Search 

```
def target_value_search(rotated_input, target_value):

    for i, n in enumerate(rotated_input):
        if n == target_value:
            return i
    return -1
```

## Good Grades and Favorite Color

```
import pandas as pd

def grades_colors(students_df: pd.DataFrame):

    return students_df[
        ((students_df.favorite_color == 'green') |
         (students_df.favorite_color == 'red'))
         & (students_df.grade > 90)]
```





