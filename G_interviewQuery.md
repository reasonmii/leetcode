
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
