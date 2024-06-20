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

















