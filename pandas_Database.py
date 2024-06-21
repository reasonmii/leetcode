import pandas as pd

# ======================================================================
# 176. Second Highest Salary
# ======================================================================

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:

    df = employee.drop_duplicates(['salary'])

    if len(df['salary'].unique()) < 2:
        return pd.DataFrame({'SecondHighestSalary':[np.NaN]})

    df.sort_values('salary', ascending=False, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.rename({'salary':'SecondHighestSalary'}, axis=1, inplace=True)
    
    return df.head(2).tail(1)

# ======================================================================
# 177. Nth Highest Salary
# ======================================================================

def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:

    df = employee.drop_duplicates(['salary'])

    if (len(df['salary'].unique()) < N) or N <= 0:
        return pd.DataFrame({'getNthHighestSalary({})'.format(N) : [np.NaN]})
    
    df.sort_values('salary', ascending=False, inplace=True)
    df.drop('id', axis=1, inplace=True)
    df.rename({'salary':'getNthHighestSalary({})'.format(N)}, axis=1, inplace=True)

    return df.head(N).tail(1)

# ======================================================================
# 180. Consecutive Numbers
# ======================================================================

def consecutive_numbers(logs: pd.DataFrame) -> pd.DataFrame:


    logs.sort_values(by='id', inplace=True)
    logs = logs[(logs.num.diff()==0)
                & (logs.num.diff().diff() == 0) 
                & ((logs.id - logs.id.shift(1)) == 1)
                & ((logs.id - logs.id.shift(2)) == 2)]

    return logs[['num']].rename(columns={'num':'ConsecutiveNums'}).drop_duplicates()

# ======================================================================
# 181. Employees Earning More Than Their Managers
# ======================================================================

import pandas as pd

def find_employees(employee: pd.DataFrame) -> pd.DataFrame:

    df = employee.merge(employee, left_on='managerId', right_on='id', suffixes=['_e', '_m'], how='inner')
    df = df[df.salary_e > df.salary_m]

    return df[['name_e']].rename(columns={'name_e':'Employee'})

# ======================================================================
# 182. Duplicate Emails
# ======================================================================

def duplicate_emails(person: pd.DataFrame) -> pd.DataFrame:

    df = person.groupby('email').size().reset_index(name='cnt')
    return df[df.cnt >= 2][['email']]

# ======================================================================
# 183. Customers Who Never Order
# ======================================================================

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:

    df = customers[~customers.id.isin(orders['customerId'])]
    return df[['name']].rename(columns={'name':'Customers'})

# ======================================================================
# 184. Department Highest Salary
# ======================================================================

def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:

    df = employee.merge(department, left_on='departmentId', right_on='id', how='left')
    df = df.rename(columns={'name_x':'Employee', 'name_y':'Department', 'salary':'Salary'})

    df['max_s'] = df.groupby('Department')['Salary'].transform('max')
    df = df[df.Salary == df.max_s]
    
    return df[['Department', 'Employee', 'Salary']]

# ======================================================================
# 185. Department Top Three Salaries
# ======================================================================

def top_three_salaries(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:

    rank = employee.groupby('departmentId').salary.rank(method='dense', ascending=False)
    df = employee[rank <= 3.0]

    df = df.merge(department, left_on='departmentId', right_on='id', how='left')
    df.rename(columns={'name_x':'Employee', 'name_y':'Department', 'salary':'Salary'}, inplace=True)

    return df[['Department', 'Employee', 'Salary']]

# ======================================================================
# 196. Delete Duplicate Emails
# ======================================================================

def delete_duplicate_emails(person: pd.DataFrame) -> None:

    person.sort_values(by='id', ascending=True, inplace=True)
    person.drop_duplicates(subset='email', keep='first', inplace=True)

# ======================================================================
# 197. Rising Temperature
# ======================================================================

def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:

    weather.sort_values(by='recordDate', ascending=True, inplace=True)

    weather['date'] = weather['recordDate'] + pd.to_timedelta(1, unit='D')
    weather['date'] = weather['date'].shift(1)

    weather['diff'] = weather.temperature.diff().fillna(0)
    
    return weather[(weather.date == weather.recordDate) & (weather['diff'] > 0)][['id']]

# ======================================================================
# 262. Trips and Users
# ======================================================================

def trips_and_users(trips: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:

    trips = trips[(trips.request_at >= '2013-10-01') & (trips.request_at <= '2013-10-03')]
    users = users[users.banned == 'No']

    df = trips.merge(users, left_on='driver_id', right_on='users_id', how='inner')
    df = df.merge(users, left_on='client_id', right_on='users_id', how='inner')

    tot = df.groupby('request_at').size().reset_index(name='tot')

    can = df[df['status'].str.startswith('cancelled')]
    can = can.groupby('request_at').size().reset_index(name='can')

    df = tot.merge(can, on='request_at', how='left').fillna(0)
    df['Cancellation Rate'] = round(df['can'] / df['tot'],2)
    
    return df[['request_at', 'Cancellation Rate']].rename(columns={'request_at':'Day'})

# ======================================================================
# 512. Game Play Analysis II
# ======================================================================

def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:

    activity['first'] = activity.groupby('player_id')['event_date'].transform('min')
    df = activity[activity['event_date'] == activity['first']]

    return df[['player_id', 'device_id']]

# ======================================================================
# 534. Game Play Analysis III
# ======================================================================

def gameplay_analysis(activity: pd.DataFrame) -> pd.DataFrame:

    activity.sort_values(by=['player_id', 'event_date'], inplace=True)
    activity['games_played_so_far'] = activity.groupby('player_id')['games_played'].cumsum()

    return activity[['player_id', 'event_date', 'games_played_so_far']]

# ======================================================================
# 550. Game Play Analysis IV
# ======================================================================

def gameplay_analysis(activity: pd.DataFrame) -> pd.DataFrame:

    activity.sort_values(by=['player_id', 'event_date'], inplace=True)
    
    activity['first'] = activity.groupby('player_id')['event_date'].transform('min')
    activity['diff_dt'] = (activity['event_date'] - activity['first']).dt.days

    pct = activity[activity['diff_dt'] == 1]['player_id'].nunique() / activity['player_id'].nunique()

    return pd.DataFrame({'fraction':[round(pct,2)]})

# ======================================================================
# 569. Median Employee Salary
# ======================================================================

def median_employee_salary(employee: pd.DataFrame) -> pd.DataFrame:

    df = employee.sort_values(by=['company', 'salary', 'id'])

    df['med'] = df.groupby('company')['salary'].transform('count')/2
    df['rk'] = df.groupby('company').cumcount() + 1

    df = df[(df.rk >= df['med']) & (df.rk <= df['med']+1)]

    return df[['id', 'company', 'salary']]

# ======================================================================
# 570. Managers with at Least 5 Direct Reports
# ======================================================================

def find_managers(employee: pd.DataFrame) -> pd.DataFrame:

    df = employee.groupby('managerId').size().reset_index(name='cnt')
    df = employee.merge(df, left_on='id', right_on='managerId', how='inner')

    return df[df.cnt >= 5][['name']]

# ======================================================================
# 571. Find Median Given Frequency of Numbers
# ======================================================================

def median_frequency(numbers: pd.DataFrame) -> pd.DataFrame:

    numbers.sort_values(by='num', inplace=True)
    nums = numbers['num'].repeat(numbers['frequency'])
    med = round(nums.median(),1)
    
    return pd.DataFrame({'median':[med]})

# ======================================================================
# 574. Winning Candidate
# ======================================================================

def winning_candidate(candidate: pd.DataFrame, vote: pd.DataFrame) -> pd.DataFrame:

    win = vote.groupby('candidateId').count().idxmax()
    return candidate[candidate.id.isin(win)][['name']]

# ======================================================================
# 577. Employee Bonus
# ======================================================================

def employee_bonus(employee: pd.DataFrame, bonus: pd.DataFrame) -> pd.DataFrame:
    
    df = employee.merge(bonus, on='empId', how='left')
    return df[(df.bonus < 1000) | (df.bonus.isna())][['name', 'bonus']]

# ======================================================================
# 578. Get Highest Answer Rate Question
# ======================================================================

def get_the_question(survey_log: pd.DataFrame) -> pd.DataFrame:

    show = survey_log[survey_log.action == 'show'].groupby('question_id').size().reset_index(name='show')
    ans = survey_log[survey_log.action == 'answer'].groupby('question_id').size().reset_index(name='ans')
    
    df = show.merge(ans, on='question_id', how='left').fillna(0)
    df['rate'] = df['ans'] / df['show']

    df.sort_values(by=['rate', 'question_id'], ascending=[False, True], inplace=True)

    return df.head(1)[['question_id']].rename(columns={'question_id':'survey_log'})

# ======================================================================
# 579. Find Cumulative Salary of an Employee
# ======================================================================

def cumulative_salary(employee: pd.DataFrame) -> pd.DataFrame:

    df = employee.sort_values(by=['id', 'month'])

    df['p1_month'] = df['month'] - df.groupby('id')['month'].shift(1).fillna(0)
    df['p2_month'] = df['month'] - df.groupby('id')['month'].shift(2).fillna(0)

    df['p1_salary'] = df.groupby('id')['salary'].shift(1).fillna(0)
    df['p2_salary'] = df.groupby('id')['salary'].shift(2).fillna(0)

    df['Salary'] = np.where(
        ((df['p1_month'] == 1) & (df['p2_month'] == 2)), df['salary'] + df['p1_salary'] + df['p2_salary'],
        np.where(df['p1_month'] == 1, df['salary'] + df['p1_salary'],
        np.where(df['p2_month'] == 2, df['salary'] + df['p2_salary'],  df['salary']
        )))

    df['max_mon'] = df.groupby('id')['month'].transform('max')
    df = df[df['month'] != df['max_mon']]

    df.sort_values(by=['id', 'month'], ascending=[True, False], inplace=True)
    return df[['id', 'month', 'Salary']]

# ======================================================================
# 580. Count Student Number in Departments
# ======================================================================

def count_students(student: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:

    df = department.merge(student, on='dept_id', how='left')
    df = df.groupby('dept_name')['student_id'].count().reset_index(name='student_number')
    df.sort_values(by=['student_number', 'dept_name'], ascending=[False, True], inplace=True)

    return df[['dept_name', 'student_number']]

# ======================================================================
# 584. Find Customer Referee
# ======================================================================

def find_customer_referee(customer: pd.DataFrame) -> pd.DataFrame:

    customer['referee_id'] = customer['referee_id'].fillna(0)
    return customer[customer.referee_id != 2][['name']]

# ======================================================================
# 586. Customer Placing the Largest Number of Orders
# ======================================================================

def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders['customer_number'].mode().to_frame()
    









