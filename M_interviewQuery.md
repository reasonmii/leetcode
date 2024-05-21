# Concept

## P-value to a Layman

How would you explain what a p-value is to someone who is not technical?
- It's the probability of observing the current data, assuming the null hypothesis is true

In a statistical test, how does a low p-value (less than 0.05) influence our decision about the null hypothesis?
- It leads us to reject the null hypothesis, because the observed data is extremely unlikely under the null hypothesis

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

## Overfit Avoidance
Let’s say that you’re training a classification model. How would you combat overfitting when building tree-based models?
- Pruning
  - Decision Trees : limit below things
    - the depth of the tree (e.g., max_depth)
    - the minimum samples required to split a node (min_samples_split)
    - the minimum samples required at a leaf node (min_samples_leaf)
- Ensemble Methods
  - Random Forests
    - Use multiple trees and aggregate their predictions.
    - Control n_estimators (number of trees), and the depth and minimum samples like in single trees
  - Gradient Boosting
    - Control n_estimators, learning rate (learning_rate), and tree-specific parameters
- Regularization
  - Random Forests
    - Use max_features to limit the number of features considered for splitting at each node
  - Gradient Boosting
    - Regularize by setting a low learning rate and subsample to use a fraction of the data for each tree
- Cross-Validation
  - Use techniques like **k-fold** cross-validation to ensure your model generalizes well to unseen data
- Feature Engineering
  - Reduce the number of features, perform feature selection, and use domain knowledge to **eliminate irrelevant or redundant features**
- Early Stopping
  - Monitor performance on a validation set and stop training when performance no longer improves
- Bootstrap Aggregating (Bagging)
  - Train multiple models on different subsets of the data and aggregate their predictions

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

# Case Study

## Analyzing Churn Behavior
Let’s say that you work for a company like Netflix. Netflix has two pricing plans: $15/month or $100/year. </br>
Let’s say an executive at the company wants you to analyze the churn behavior of users that subscribe to either plan. </br>
What kinds of metrics / graphs / models would you build to help give the executive an over-arching view of how the subscriptions are performing? </br>

Metrics
- Customer Demographic : age, gender, location
- Tenure : the length of time a customer has been with the service
- Revenue
  - Average Revenue Per User (ARPU) for each plan
  - Customer Lifetime Value : the projected revenue a customer will generate during their lifetime with the company
- Past Number of Subscriptions
- Percentage Revenue Change Year-over-Year : how revenue from the customer has changed compared to the previous year
- Number of shifts from monthly plan to yearly or vice versa
- Time since last subscription
- Churn rate : monthly, annual
- Retention rate : monthly, annual
- Customer Acquisition Cost (CAC) : cost associated with acquiring a new customer for each pricing plan
- Net Promoter Score (NPS) : a measure of customer satisfaction and loyalty

Graphs
- churn rate over time
- retention rate over time
- survival analysis : the probability of subscribers **remaining active** over time for each pricing plan
- cohort analysis (집단) : heatmaps showing retention rates for cohorts of users who subscribed in the same month/year, segmented by pricing plan
- Customer Lifetime Value (CLV) Distribution : histograms or box plots showing the distribution of CLV for each pricing plan
- ARPU over time
- Seasonality of Loss : identifying any seasonal patterns in when customers are more likely to cancel their subscriptions

Models
- Churn Prediction Model : Logistic Regression, Random Forest, Gradient Boosting
- Survival Analysis Model
- CLV Prediction Model

## Rebalance Probabilities
Suppose we are training a binary classification algorithm. </br>
The outcome is imbalanced, with 99.8% of individuals in our sample having an outcome value of 0, and 0.2% having an outcome value of 1. </br>
To build a model, we down-sample the training data using a random sample of 1% of the individuals who have an outcome of 0 and keeping all individuals with an outcome of 1. </br>
Now we train the model on the smaller sample and build a binary classifier that predicts the probability of an individual with a value of 1. </br>
How would we then adjust our output probabilities to use this model on the total, imbalanced population?

We can use techniques like **probability calibration** to adjust the output probabilities to better reflect the true probabilities in the total population.
- One common method for probability calibration is **Platt scaling** (Platt calibration)
- It calibrates the output probabilities of a binary classifier
- Method
  - Treat the uncalibrated probabilities as the independent variable and the true binary labels as the dependent variable.
  - Fit a logistic regression model to this data.
  - The logistic regression model learns to map the uncalibrated probabilities to the true probabilities.
  - Use the model to alibrate the probabilities produced by your binary classifier.

Or we can use the formula
- https://www.knime.com/blog/correcting-predicted-class-probabilities-in-imbalanced-datasets
- $P+ = (P+ \cdot P_0+/P_t+) / (P+ (P_0+/P_t+) + (1-P+) (1-P_0+)/(1-P_t+))$
- $P_+'$ : the corrected positive class probability
- $P+$ : positive class probability predicted by the model
- $P_0+$ : the proportion of the positive class in the original data
- $P_t+$ : the proportion of the positive class in the training data

## Score Based on Review
Let’s say you’re an ML engineer at Netflix. </br>
You have access to reviews of 10K movies. Each review contains multiple sentences along with a score ranging from 1 to 10. </br>
How would you design an **ML system to predict the movie score based on the review text**?

Goal : Predict the score (ranging from 1 to 10) of a movie based on its review text.
- Data Cleaning
  - Remove any duplicate or irrelevant reviews.
  - Handle missing values, if any.
  - Normalize the text (e.g., lowercasing, removing punctuation, stop words, etc.).
- Exploratory Data Analysis (EDA)
  - Analyze the distribution of review scores.
  - **Identify common words** and phrases in high vs. low-rated reviews.
  - Use word clouds, histograms, and other visualizations to understand the text data.
- Feature Engineering
  - Text Representation : Convert review text into numerical features.
    - **Bag of Words** (BoW): Represent text using **word frequency.**
    - **TF-IDF** (Term Frequency-Inverse Document Frequency): Adjust word frequency by how often they appear in all reviews.
    - **Word Embeddings :** Use pre-trained embeddings (e.g., Word2Vec, GloVe) or transformer-based models (e.g., BERT) for contextual word representations.
  - Additional Features
    - Sentence length.
    - Sentiment analysis scores.
    - Named entity recognition (NER) for extracting specific details (e.g., movie names, actors).
- Model Selection
  - Baseline Models : Start with simpler models like Linear Regression or Ridge Regression for initial benchmarks.
  - Advanced Models
    - Explore more complex models such as Random Forest, Gradient Boosting, or Neural Networks.
    - Specifically, consider using Recurrent Neural Networks **(RNNs)** or Transformers (e.g., **BERT**) for their ability to **handle sequential text data effectively.**
- Model Training and Evaluation
  - Data Splitting : training, validation, and test sets (e.g., 70% train, 15% validation, 15% test).
  - Training
    - Train the selected models on the training dataset.
    - Use cross-validation to tune hyperparameters and avoid overfitting.
  - Evaluation Metrics
    - Mean Squared Error (MSE) for regression accuracy.
    - R-squared (R²) for explained variance.
    - Consider additional metrics like Mean Absolute Error (MAE) for robustness.
- Model Interpretation and Explainability
  - Feature Importance : understand which words or phrases are most influential in predicting the score
    - SHAP (SHapley Additive exPlanations) values
    - LIME (Local Interpretable Model-agnostic Explanations)
  - Model Insights
    - Generate insights on common characteristics of reviews associated with different scores.
- Model Deployment
  - Deploy the model as an API service within the Netflix platform.
  - Ensure the model integrates seamlessly with existing recommendation systems.
- Monitoring and Maintenance
  - Continuously monitor model performance using real-time feedback.
  - Regularly retrain the model with new review data to maintain accuracy and relevance.
  - Implement alerting mechanisms for significant performance drops.
- Ethical Considerations and Bias Mitigation
  - Bias Detection: Regularly assess the model for biases (e.g., demographic biases) and ensure fair predictions.
  - Transparency: Maintain transparency with users regarding how their reviews are used and the purpose of the predictions.

## Keyword Bidding
Let’s say you’re working on keyword bidding optimization. </br>
You’re given a dataset with two columns. </br>
One column contains the keywords that are being bid against, and the other column contains the price that’s being paid for those keywords. </br>
Given this dataset, how would you build a model to bid on a new unseen keyword?

- Data Preprocessing
  - Text Processing
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - word embeddings (Word2Vec, GloVe, BERT)
  - Normalization : Normalize the price
- Feature Engineering
  - Keyword Features
  - Additional Features : consider adding features such as keyword length, frequency, or other relevant metrics
- Split the Dataset into training and test set
- Model Selection : Linear regression, Decision Trees, Random Forests, Gradient Boosting, Neural Networks
- Model Training
- Model Evaluation : Mean Squared Error, Mean Absolute Error, R-squared
- Prediction for New Keywords

## Spanish Scrabble
Let’s say you have to build scrabble for Spanish users. </br>
Assuming that you don’t know any Spanish, how would you approach assigning each letter a point value? </br>

- It requires a structured and data-driven approach
- Frequency Analysis
  - Collect Spanish Text Data (books, newspapers, websites, and other sources of written Spanish)
  - Calculate Letter Frequencies : **More frequent** letters should generally have **lower point** values, while **less frequent** letters should have **higher point** values.
- Existing Spanish Games
  - Research Spanish Word Games : Analyze the point values assigned in these games.
  - Compare with English Scrabble : apply a similar rationale
- Statistical Modeling
  - Fit a Model : model the relationship between letter frequency and point value.
  - Assign Point Values: Ensure that the distribution of points creates a balanced game with an appropriate level of difficulty and competition.
- Iterative Testing
  - Prototype and Test : Create a prototype of the game with the initial point values and test it with Spanish-speaking players.
  - Gather Feedback : Collect feedback on the balance and playability of the game.
  - Refine Point Values: Adjust the point values based on feedback and further analysis.
- Consult Linguistic Experts
  - Collaborate with Spanish Linguists : Work with experts in the Spanish language to refine your approach and ensure cultural and linguistic appropriateness.

## Video Game Respawn Model
How would you build a model or algorithm to generate respawn locations for an online third person shooter game like Halo?
When designing an algorithm to generate respawn locations in an online game, what aspects must be considered to ensure long-term player engagement?

## Google Docs Autosave System
Let’s say that you work with the Google Docs team. </br>
Currently, Google Docs offers an autosave capability that seamlessly saves user modifications to documents directly to cloud storage in real-time. </br>
However, the autosave function introduces performance bottlenecks within our data processing pipeline, mainly due to slow disk writes. </br>
How could we enhance the autosave feature to boost data processing throughput and reduce save latency? </br>
State any assumptions upfront.

Assumptions
- High-Frequency Saves: Users frequently modify documents, leading to frequent autosave requests.
- Network Latency: The system is cloud-based, and network latency may contribute to overall save latency.
- Disk I/O Bottlenecks: The primary bottleneck is disk I/O during the autosave process.
- Consistency Requirements: The system requires strong consistency to ensure that user modifications are not lost.
- Scalability: The solution needs to scale with the number of users and documents

Solutions
- **Batch** Writes: Collect multiple changes and save them together at intervals.
  - Instead of saving every single modification in real-time, batch multiple changes and save them periodically.
- Asynchronous I/O: Use background processes for saving to prevent blocking.
  - Implement a write-ahead log to quickly record changes to disk before processing them further.
- Incremental Saving: only write the changes since the last save instead of rewriting the entire document.
- **Caching**: Use in-memory caching for quick writes before saving to disk.
  - temporarily store modifications
- Compression: Compress changes to reduce data size before saving.
  - Apply compression techniques to reduce the amount of data being written to disk.
  - particularly useful if the changes being saved are small but frequent.

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

## Disease Testing Probability

Bob takes a test and **tests positive** for the disease. </br>
Bob was close to six other friends they all take the same test and end up testing negative. </br>
The test has a 1% false positive rate and a 15% false negative rate. </br>
What’s the percent chance that Bob was **actually negative** for the disease? P(No Disease) </br>

**Bayes Theorem**
- FPR : P(Test Positive | No Disease) = 0.01
  - P(Test Negative | No Disease) = 1 - 0.01 = 0.99
- FNR : P(Test Negative | Disease) = 0.15
  - P(Test Positive | Disease) = 1 - 0.15 = 0.85
- P(Test Positive) = 0.01 * (1 - P(D)) + 0.85 * P(D) = 0.84 * P(D) + 0.01 = 0.84p + 0.01
- P(No Disease | Test Positive) = 0.01(1-p) / (0.84p + 0.01)
- Let's assume that p = 0.05 (5%)
  - P(Test Positive) = 0.042 + 0.01 = 0.052
  - P(No Disease | Test Positive) = 0.01 (0.95) / 0.052 = 0.1827

## Tower of Hanoi Algorithm
The Tower of Hanoi is a mathematical puzzle that consists of three rods and several discs of different sizes that can slide onto any rod. </br>
The puzzle starts with all the discs stacked on a single rod, the smallest disc at the top, making a cone shape. </br>
The object of the puzzle is to move the entire stack to another rod, obeying these rules:
- Only one disc can be moved at a time.
- Each move consists of taking the upper disc from one of the stacks and placing it on top of another stack or an empty rod.
- No larger disc may be placed on top of a smaller disc.
With these rules in mind, how would you approach solving the puzzle if there are three discs on the leftmost rod and your goal is to move them all to the rightmost rod?

Step-by-Step Solution
- Label the rods: Source rod: A, Auxiliary rod: B. Destination rod: C
- Move 2 discs from A to B using C as auxiliary
  - Move disc 1 from A to C.
  - Move disc 2 from A to B.
  - Move disc 1 from C to B.
- Move the 3rd disc from A to C.
- Move 2 discs from B to C using A as auxiliary
  - Move disc 1 from B to A.
  - Move disc 2 from B to C.
  - Move disc 1 from A to C.

Pseudocode
```
Function TowerOfHanoi(n, source, auxiliary, destination):
    if n == 1:
        Print "Move disc 1 from " + source + " to " + destination
        return
    TowerOfHanoi(n-1, source, destination, auxiliary)
    Print "Move disc " + n + " from " + source + " to " + destination
    TowerOfHanoi(n-1, auxiliary, source, destination)
```

## Google Earth Storage
Estimate the cost of storing Google Earth photos each year
- Lets say google earth is taking photos of every 1x1m block on earth.
- Each photo is 1 MB is a compressed version.
- Now diameter of earth is 12.742M m. So radius is ~6.3M m.
- Assuming earth is a perfect sphere, surface area of earth would be : $4/3 \pi r^3$
- Thus, the total size of pic will be : $4/3 \pi (6300000)^3$
- Assuming cost of storing 1MB photo is 5 cents, total cost will be : $5 \times 4/3 \pi (6300000)^3$

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

## Employee Project Budgets
- Top five most expensive projects by budget to employee count ratio
- Exclude projects with 0 employees

```
select p.title
     , p.budget / e.cnt budget_per_employee
from projects p
inner join (select project_id
                 , count(*) cnt
                 # , count(distinct employee_id) cnt # when there's duplicate
            from employee_projects
            group by 1
            ) e on e.project_id = p.id
order by budget_per_employee desc
limit 5
```

# Python

## Precision and Recall

```
def precision_recall(P):
    
    tp = P[0][0]
    fp = P[1][0]
    fn = P[0][1]

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    return (pre, rec)
```

## Swap Variables

```
def swap_values(numbers):

  numbers['a'], numbers['b'] = numbers['b'], numbers['a']
  return numbers
```

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

## Greatest Common Denominator

```
def gcd(numbers):

    min_v = min([num for num in numbers if num > 0])

    for n in range(min_v,0,-1):
        if sum([num % n for num in numbers]) == 0:
            return n
```

## N N Grid Traversal

```
def traverse_count(n):

    grid = [[1 for _ in range(n)] for _ in range(n)]
    
    for row in range(n):
        for col in range(n):
            if row != 0 and col != 0:
                grid[row][col] = grid[row-1][col] + grid[row][col-1]
    
    return grid[n-1][n-1]
```

## The Brackets Problem

```
def is_balanced(string: str) -> bool:
    
    rst = []
    for ch in string:
        if ch in '([{':
            rst.append(ch)
        else:
            if ch == ')' and rst[-1] != '(' or \
                ch == ']' and rst[-1] != '[' or \
                ch == '}' and rst[-1] != '{':
                return False
            else:
                rst.pop()

    return len(rst) == 0
```

## String Palindromes

```
def is_palindrome(word):
    # return word == word[::-1]
    i, j = 0, len(word)-1
    while i < j:

        if word[i] != word[j]:
            return False

        i += 1
        j -= 1

    return True
```

## Length Of Longest Palindrome

```
def longest_palindrome(s):

    dic = {}    
    for ch in s:
        if ch in dic:
            dic[ch] += 1
        else:
            dic[ch] = 1
    
    odd_max = 0
    cnt = 0

    for k, v in dic.items():
        if v % 2 == 0:
            cnt += v
        else:
            if odd_max < v:
                odd_max = v

    return cnt + odd_max
```

## Find Bigrams

```
def find_bigrams(sentence):

    words = sentence.lower().split()
    rst = []

    # list(zip(words, words[1:]))
    for i in range(1, len(words)):
        rst.append((words[i-1], words[i]))

    return rst
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

## Binary Tree Conversion

```
def convert_to_bst(sorted_list: list) -> TreeNode:
    
    if not sorted_list:
        return None

    mid = len(sorted_list) // 2
    root = TreeNode(sorted_list[mid])

    root.left = convert_to_bst(sorted_list[:mid])
    root.right = convert_to_bst(sorted_list[mid+1:])

    return root
```










