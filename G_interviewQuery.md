
# Statistics

## Best Measure
Say you have a dataset and you are asked to analyze it by looking at some metrics like the mean and median.

When would you use one measure over the other?
- use the mean for symmetric, normally distributed data
- the median for skewed or ordinal data

How do you calculate the confidence interval of each measure?
- Confidence Interval for the Mean
  - Assuming Normality
    - standard error of the mean : $\sigma / \sqrt{n}$
    - 95% confidence interval : z-value is approximately 1.96
    - $CI = \bar{x} \pm (1.96 \times \sigma / \sqrt{n}$
  - Without Assuming Normality
    - If the sample size is small or the distribution is not normal, we can use the **t-distribution**
    - $CI = \bar{x} \pm (t \times \sigma / \sqrt{n-1}$
- Confidence Interval for the Median
  - Bootstrap Method
    - Resample the dataset with replacement many times, each time calculating the median
    - Use the distribution of these bootstrap medians to determine the confidence interval
    - For a 95% confidence interval, take the 2.5th and 97.5th percentiles of the bootstrap medians
  - Order Statistics (for large samples)
    - The confidence interval can be approximated using **order statistics**
    - For a 95% confidence interval, find the ranks corresponding to the 2.5th and 97.5th percentiles of the sorted data.

## Distribution of 2X - Y

Given that $X$ and $Y$ are independent random variables with normal distributions,
and the corresponding distributions are $X\sim \mathcal{N}(3, 4)$ and $Y \sim \mathcal{N}(1, 4)$,
what is the mean of the distribution of $2X - Y$?
- $E(2X - Y) = 2 \times 3 - 1 = 5$
- $Var(2X - Y) = 2^2 Var(X) + (-1)^2 Var(Y) = 16 + 4 = 20$

## Mutated Offspring

An animal appears normal if it has two normal genes or one normal and one mutated gene.
If it has two mutated genes, it appears mutated.
Any given animal has a 50% chance to contribute either of its genes to its offspring.

Animals A and B are parents of C and D. C and D are parents of E.
A and B both have one normal and one mutated gene.
We know C and D both appear normal.

What is the probability that **D has one normal and one mutated** gene given that E appears normal?

- total 4 cases : 2 nor, nor * mut, mut * nor, 2 mut
- P(normal)
  - P(nor * mut) = 1/2
  - P(2 nor) = 1/4
- P(E = nor)
  - P(C = 2 nor) * P(D = 2 nor) = 1/16 -> E always normal
  - P(C = 2 nor) * P(D = nor & mut) = 1/8 -> E always normal
  - P(C = nor & mut) * P(D = 2 nor) = 1/8 -> E always normal
  - P(C = nor & mut) * P(D = nor & mut) = 1/4 -> E to be normal : 1/4 * 3/4 = 3/16
- P(C = nor, D = nor, E = nor) = 1/16 + 1/8 + 1/8 + 3/16 = 1/2
- P(D = nor & mut | E = nor) = (1/8 + 3/16) / 1/2 = 5/8

# Python

You’re given two dataframes.
One contains information about addresses and the other contains relationships between various cities and states.
Write a function complete_address to create a single dataframe with complete addresses in the format of street, city, state, zip code.

```
import pandas as pd

def complete_address(df_addresses: pd.DataFrame, df_cities: pd.DataFrame):
    df_addresses[['street', 'city', 'zip code']] = df_addresses['address'].str.split(', ', expand=True) # expand the result into a DataFrame
    df = df_addresses.merge(df_cities, on='city')
    df['address'] = df[['street', 'city', 'state', 'zip code']].apply(
        lambda x: ', '.join(x), axis=1
    )
    
    return df[['address']]
```

# SQL

Given the employees and departments table, write a query to get the top 3 highest employee salaries by department.
If the department contains less that 3 employees, the top 2 or the top 1 highest salaries should be listed (assume that each department has at least 1 employee). 

Note: The output should include the full name of the employee in one column, the department name, and the salary.
The output should be sorted by department name in ascending order and salary in descending order. 

```
with t as (
    select concat(e.first_name, ' ', e.last_name) employee_name
        , d.name department_name
        , e.salary
        , rank()over(partition by d.name order by e.salary desc) rk
    from departments d
    left join employees e on d.id = e.department_id
)
select t.employee_name
     , t.department_name
     , t.salary
from t
where t.rk <= 3
order by t.department_name, t.salary desc
```

# Case Study

## Job Recommendation

Let’s say that you’re working on a job recommendation engine.
You have access to all user Linkedin profiles, a list of jobs each user applied to, and answers to questions that the user filled in about their job search.
Using this information, how would you build a job recommendation feed?

Questions
- How big is our Job Database size? How big is our client size?
- Are we dealing with millions of jobs in that case we need to understand what kind of ML algo we need to fit the data correctly?
- How fast do we need our response?
  - this will determine if we need to use any funneling technique to handle large data
- Are we looking for diversity, speed, the accuracy of the recommendation product?
- What is the Service Level Agreement to this app?

PROBLEM Statement
- Given a User and Context predict the probability of a Job being recommended to a User.
  - This could be looked at like a **Binary Classification** Problem
    - 1 : show to user
    - 0 : do not show it to user
- The problem can be broken down into two steps
  - Candidate Selection
    - Out of the entire DB of jobs in our system, we need to select the subset of jobs based on User profile info which will be relevant to the User
    - Candidate Ranking : Rank the jobs according to most relevant to Least

Metrics
- Used to test various Models : Precision, Accuracy, F-1 Score
- Revenue Generated
  - Session Time of the user
  - Click-Through-rate = Clicks/ jobs Recommended
  - Apply-Rate = jobs Applied/ Jobs Recommended
  - Company Success through our platform = total Candidates Hired by Our Platform / total job postings made
- There could be a thumbs up and down button on the UI to ask for User Feedback based on what we are recommending
- Jobs ignored these will be the jobs which the user was recommended and did not apply to
  - These will count as Negative Training Data.

Architecture
- Feature Engineering

Who are the actors in the system?
- Advertisers/ Companies
  - • Company Size • Company Ratings • Company Reviews • Locations • Industry (Healthcare, Tech, Fashion etc.)
- Job Seekers/ Users
  - a. Age b. Years of Experience c. Current Job Title d. Location e. Language f. Degree g. Certifications h. Skillsets i. Background
- Job
  - Job Title (Very important when user queries) b. Job Description c. Min Years of Ex d. Compensation e. Team Size f. Location 
- User-Application Embedding historical
- User-Job behavior

Model Training
- Collaborative Filtering
- We look at the Users like the current user and recommend jobs that other users have also applied to.

KNearest Neighbors
- We identify the K Nearest neighbor to the Current user based on their background, skills, years of experience and then recommend jobs to the user which the user has not yet applied to.
- There can be a matrix of J(jobs) x U(users).
  - This will be a binary sparse matrix with 1/0/ empty in a cell.
  - If it is 1 then the user has applied, 0 means not applied or did not like, empty means have not yet seen or applied.
  - Given the empty cells for a user U we want to predict of that cell value will be 0/1 based on the other users similar to U.

Matrix Factorization
- The above method can be computationally very cumbersome as it will be very big.
- Hence, we need to perform Matrix Factorization to allow the dimensions to be reduced to Latent Dimension.

Content Based Filtering
- This uses the Textual Information in the Job Description along with the User’s information to extract TF-IDF details for understanding similarity between the Job and the User.

- Comprehensive overview of Recommender systems at different top tech companies : https://towardsdatascience.com/recommender-systems-in-practice-cef9033bb23a

### What problem can arise in job recommendation systems with regard to initial recommendations?
- Initial recommendations may introduce bias based on predetermined algorithms
