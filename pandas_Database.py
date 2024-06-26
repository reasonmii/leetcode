import pandas as pd

# size() returns a Series with a MultiIndex composed of the levels student_id and subject_name, and the values are the counts. 
# By default, these counts don't have a column name,
# so reset_index not only turns the index back into regular columns
# but also allows you to name the column of counts using the name parameter. 

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
    
# ======================================================================
# 597. Friend Requests I: Overall Acceptance Rate
# ======================================================================

def acceptance_rate(friend_request: pd.DataFrame, request_accepted: pd.DataFrame) -> pd.DataFrame:
    
    request = len(friend_request[['sender_id','send_to_id']].drop_duplicates())
    accept = len(request_accepted[['requester_id', 'accepter_id']].drop_duplicates())
    rate = 0 if request == 0 else accept / request

    return pd.DataFrame({'accept_rate': [rate]}).round(2)

# ======================================================================
# 601. Human Traffic of Stadium ###
# ======================================================================

def human_traffic(stadium: pd.DataFrame) -> pd.DataFrame:

    stadium = stadium[stadium.people >= 100]

    # diff() : compare the current id and the id of the previous row
    # diff().shift(1) : compare the previous row id and the id before the previous row
    stadium['flag'] = ((stadium['id'].diff() == 1) & (stadium['id'].diff().shift(1) == 1)) # null, false, true

    stadium = stadium[(stadium['flag']) | (stadium['flag'].shift(-1)) | (stadium['flag'].shift(-2))]

    # remove the column 'flag'
    return stadium.drop(columns='flag').sort_values(by='visit_date')
    
# ======================================================================
# 602. Friend Requests II: Who Has the Most Friends ###
# ======================================================================

def most_friends(request_accepted: pd.DataFrame) -> pd.DataFrame:

    df = pd.concat([request_accepted['requester_id'], request_accepted['accepter_id']]).to_frame('id')
    df = df.groupby('id').size().reset_index(name='num')
    df.sort_values(by='num', ascending=False, inplace=True)

    return df.head(1)

# ======================================================================
# 608. Tree Node
# ======================================================================

def tree_node(tree: pd.DataFrame) -> pd.DataFrame:

    tree['type'] = np.where(tree['p_id'].isna(), 'Root',
                   np.where(tree['id'].isin(tree['p_id']), 'Inner', 'Leaf'))

    return tree[['id', 'type']]

# ======================================================================
# 610. Triangle Judgement ###
# ======================================================================

def triangle_judgement(triangle: pd.DataFrame) -> pd.DataFrame:
    
    triangle['triangle'] = triangle.apply(
        lambda t: 'Yes' if ((t.x + t.y > t.z) and (t.x + t.z > t.y) and (t.y + t.z > t.x))
        else 'No', axis=1)

    return triangle

# ======================================================================
# 612. Shortest Distance in a Plane ###
# ======================================================================

def shortest_distance(point2_d: pd.DataFrame) -> pd.DataFrame:

    df = point2_d.merge(point2_d, how='cross')
    df = df[(df.x_x != df.x_y) | (df.y_x != df.y_y)]

    df['dist'] = round(((df['x_x'] - df['x_y'])**2 + (df['y_x'] - df['y_y'])**2)**0.5, 2)
    return pd.DataFrame({'shortest':[df['dist'].min()]})

# ======================================================================
# 613. Shortest Distance in a Line
# ======================================================================

def shortest_distance(point: pd.DataFrame) -> pd.DataFrame:

    point.sort_values('x', inplace=True)
    short = point.x.diff().min()
    return pd.DataFrame({'shortest':[short]})

# ======================================================================
# 615. Average Salary: Departments VS Company
# ======================================================================

def average_salary(salary: pd.DataFrame, employee: pd.DataFrame) -> pd.DataFrame:

    salary['pay_month'] = salary['pay_date'].dt.strftime("%Y-%m") ###
    
    mon = salary.groupby('pay_month').agg({'amount':'mean'}).reset_index()

    dept = salary.merge(employee, on='employee_id', how='left')
    dept = dept.groupby(['pay_month', 'department_id']).agg({'amount':'mean'}).reset_index()
    dept = dept.merge(mon, on='pay_month', how='left')

    dept['comparison'] = np.where(dept['amount_x'] == dept['amount_y'], 'same',
                         np.where(dept['amount_x'] > dept['amount_y'], 'higher', 'lower'))

    return dept[['pay_month', 'department_id', 'comparison']]

# ======================================================================
# 618. Students Report By Geography
# ======================================================================

def geography_report(student: pd.DataFrame) -> pd.DataFrame:

    am = student[student['continent'] == 'America'][['name']]
    am.sort_values(by='name', inplace=True)
    am = am.rename(columns={'name':'America'}).reset_index(drop=True)

    ai = student[student['continent'] == 'Asia'][['name']]
    ai.sort_values(by='name', inplace=True)
    ai = ai.rename(columns={'name':'Asia'}).reset_index(drop=True)

    eu = student[student['continent'] == 'Europe'][['name']]
    eu.sort_values(by='name', inplace=True)
    eu = eu.rename(columns={'name':'Europe'}).reset_index(drop=True)

    return pd.concat([am, ai, eu], axis=1)

# ======================================================================
# 619. Biggest Single Number ###
# ======================================================================

def biggest_single_number(my_numbers: pd.DataFrame) -> pd.DataFrame:

    # keep=False : all duplicates will be removed,
    return my_numbers.drop_duplicates(keep=False).max().to_frame(name='num')

# ======================================================================
# 626. Exchange Seats
# ======================================================================

def exchange_seats(seat: pd.DataFrame) -> pd.DataFrame:

    id = list(range(1, len(seat)+1))

    for i in range(1, len(seat), 2):
        id[i], id[i-1] = id[i-1], id[i]

    seat['id'] = id
    return seat.sort_values('id')

# ======================================================================
# 1069. Product Sales Analysis II
# ======================================================================

def sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    
    df = sales.merge(product, on='product_id', how='left')
    df = df.groupby('product_id').agg(total_quantity=('quantity', 'sum')).reset_index()
    return df
    
# ======================================================================
# 1070. Product Sales Analysis III
# ======================================================================

def sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:

    sales['rk'] = sales.groupby('product_id')['year'].rank(method='dense', ascending=True)
    sales.rename(columns={'year':'first_year'}, inplace=True)
    return sales[sales.rk == 1][['product_id', 'first_year', 'quantity', 'price']]

# ======================================================================
# 1076. Project Employees II
# ======================================================================

def project_employees_ii(project: pd.DataFrame, employee: pd.DataFrame) -> pd.DataFrame:

    df = project.groupby('project_id').size().reset_index(name='cnt')
    max_v = df['cnt'].max()

    return df[df['cnt'] == max_v][['project_id']]

# ======================================================================
# 1083. Sales Analysis II
# ======================================================================

def sales_analysis(product: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:

    df = sales.merge(product, on='product_id', how='left')

    df = df.groupby('buyer_id').agg(
        s8 = ('product_name', lambda x:(x == 'S8').sum()),
        ip = ('product_name', lambda x:(x == 'iPhone').sum())
    ).reset_index()

    return df[(df.s8 > 0) & (df.ip == 0)][['buyer_id']]

# ======================================================================
# 1084. Sales Analysis III
# ======================================================================

def sales_analysis(product: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:

    st = pd.to_datetime('2019-01-01')
    ed = pd.to_datetime('2019-03-31')

    df = sales.groupby('product_id').filter(
        lambda x: min(x['sale_date']) >= st and max(x['sale_date']) <= ed
    )

    df = df.drop_duplicates(subset='product_id')
    
    df = df.merge(product, on='product_id', how='left')

    return df[['product_id', 'product_name']]

# ======================================================================
# 1097. Game Play Analysis V ###
# ======================================================================

def gameplay_analysis(activity: pd.DataFrame) -> pd.DataFrame:

    activity['install_dt'] = activity.groupby('player_id')['event_date'].transform('min')
    activity['log'] = activity['install_dt'] + pd.DateOffset(1)

    ins = activity[activity.event_date == activity.install_dt]
    ins = ins.groupby('event_date').size().reset_index(name='installs')

    log = activity[activity.event_date == activity.log]
    log = log.groupby('install_dt').size().reset_index(name='log')

    df = ins.merge(log, left_on='event_date', right_on='install_dt', how='left').reset_index().fillna(0)
    df['Day1_retention'] = (df['log'] / df['installs'] * 100 + 0.5).astype(int) / 100 ###

    return df[['event_date', 'installs', 'Day1_retention']].rename(columns={'event_date':'install_dt'})

# ======================================================================
# 1098. Unpopular Books
# ======================================================================

def unpopular_books(books: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:

    df = books[books.available_from < pd.to_datetime('2019-05-23')]

    od = orders[orders.dispatch_date > pd.to_datetime('2018-06-23')] ###
    od = od.groupby('book_id').agg({'quantity':'sum'}).reset_index()
    od = od[od.quantity >= 10]['book_id'].unique() ###

    return df[~df['book_id'].isin(od)][['book_id', 'name']]

# ======================================================================
# 1107. New Users Daily Count
# ======================================================================

def new_users_daily_count(traffic: pd.DataFrame) -> pd.DataFrame:

    traffic.sort_values(['user_id', 'activity_date'], inplace=True)
    df = traffic[traffic.activity=='login']
    df = df.groupby('user_id').head(1) ###
    df = df[df.activity_date >= pd.to_datetime('2019-06-30') - pd.Timedelta(days=90)] ##
    df = df.groupby('activity_date').size().reset_index(name='user_count')

    return df.rename(columns={'activity_date':'login_date'})

# ======================================================================
# 1112. Highest Grade For Each Student
# ======================================================================

def highest_grade(enrollments: pd.DataFrame) -> pd.DataFrame:

    df = enrollments.sort_values(['student_id', 'grade', 'course_id'], ascending=[True, False, True])
    df['rk'] = df.groupby('student_id')['grade'].rank(method='dense', ascending=False)
    df = df[df.rk == 1].groupby('student_id').head(1) ###

    return df[['student_id', 'course_id', 'grade']]

# ======================================================================
# 1127. User Purchase Platform
# ======================================================================

def user_purchase(spending: pd.DataFrame) -> pd.DataFrame:

    dt = pd.DataFrame({'spend_date': spending['spend_date'].unique()})
    pl = pd.DataFrame({'platform': ['mobile', 'desktop', 'both']})
    idx = dt.merge(pl, how='cross')

    mo = spending[spending.platform == 'mobile']
    de = spending[spending.platform == 'desktop']
    bo = de.merge(mo, on=['user_id', 'spend_date'], how='inner', suffixes=['_m', '_d'])

    mo2 = mo.merge(de, on=['user_id', 'spend_date'], how='left', suffixes=['', '_d'])
    mo2 = mo2[mo2['platform_d'].isnull()]
    mo2 = mo2.groupby(['spend_date']).agg(
        total_amount=('amount', 'sum'),
        total_users=('user_id', 'count')
    ).reset_index()
    mo2['platform'] = 'mobile'

    de2 = de.merge(mo, on=['user_id', 'spend_date'], how='left', suffixes=['', '_m'])
    de2 = de2[de2['platform_m'].isnull()]
    de2 = de2.groupby(['spend_date']).agg(
        total_amount=('amount', 'sum'),
        total_users=('user_id', 'count')
    ).reset_index()
    de2['platform'] = 'desktop'

    bo['amount'] = bo['amount_m'] + bo['amount_d']
    bo = bo.groupby(['spend_date']).agg(
        total_amount=('amount', 'sum'),
        total_users=('user_id', 'count')
    ).reset_index()
    bo['platform'] = 'both'

    df = pd.concat([de2, mo2, bo], axis=0)
    return idx.merge(df, on=['spend_date', 'platform'], how='left').fillna(0)

# ======================================================================
# 1132. Reported Posts II
# ======================================================================

def reported_posts(actions: pd.DataFrame, removals: pd.DataFrame) -> pd.DataFrame:

    df = actions[actions.extra == 'spam'].drop_duplicates(['post_id', 'action_date'])
    df = df.merge(removals, on='post_id', how='left')

    ###
    # count : the number of non-null values
    # size : all values (non-null + null)
    df = df.groupby('action_date').agg(
        rem = ('remove_date', 'count'),
        tot = ('remove_date', 'size')
    )

    df['average_daily_percent'] = df['rem'] / df['tot'] * 100

    # need [] -> ex) ['mean'] 
    avg = df.agg({'average_daily_percent':['mean']}).round(2)

    return avg

# ======================================================================
# 1142. User Activity for the Past 30 Days II
# ======================================================================

def user_activity(activity: pd.DataFrame) -> pd.DataFrame:

    e_day = '2019-07-27'
    s_day = '2019-06-28'
    df = activity[(activity['activity_date'] >= s_day) & (activity['activity_date'] <= e_day)]

    df = df.groupby('user_id')['session_id'].nunique()
    
    avg_val = round(df.mean(),2)
    avg_val = avg_val if not pd.isna(avg_val) else 0.00 ###

    return pd.DataFrame({'average_sessions_per_user':[avg_val]})

# ======================================================================
# 1159. Market Analysis II
# ======================================================================

def market_analysis(users: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:

    df = orders.merge(users[['user_id', 'favorite_brand']], left_on='seller_id', right_on='user_id', how='left')
    df = df.merge(items, on='item_id', how='left')

    df['rank'] =  df.groupby('seller_id')['order_date'].rank()

    yes = df[(df.favorite_brand == df.item_brand) & (df['rank'] == 2)]['seller_id'].to_list() ###

    df = users[['user_id']].rename(columns={'user_id':'seller_id'})
    df['2nd_item_fav_brand'] = df['seller_id'].apply(lambda x: 'yes' if x in yes else 'no') ###

    return df

# ======================================================================
# 1179. Reformat Department Table ###
# ======================================================================

def reformat_table(department: pd.DataFrame) -> pd.DataFrame:

    df = department.pivot(index='id', columns='month', values='revenue')
    df = df.reindex(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    df.rename(columns=lambda x: x+'_Revenue', inplace=True)
    df.reset_index(inplace=True)
    return df

# ======================================================================
# 1193. Monthly Transactions I ###
# ======================================================================

def monthly_transactions(transactions: pd.DataFrame) -> pd.DataFrame:

    transactions['month'] = transactions['trans_date'].dt.strftime('%Y-%m')

    transactions['app'] = np.nan
    transactions.loc[transactions.state == 'approved', 'app'] = transactions['amount']

    df = transactions.groupby(['month', 'country'], dropna=False).agg(
        trans_count=('id', 'count'),
        approved_count=('app', 'count'),
        trans_total_amount=('amount', 'sum'),
        approved_total_amount=('app', 'sum')
    ).reset_index()

    return df    

# ======================================================================
# 1194. Tournament Winners
# ======================================================================

def tournament_winners(players: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:

    fi = matches.groupby('first_player')['first_score'].sum() # series
    se = matches.groupby('second_player')['second_score'].sum() # series

    df = fi.add(se, fill_value=0).reset_index(name='score') ###
    df = df.merge(players, left_on='index', right_on='player_id', how='left')

    df['max_v'] = df.groupby('group_id')['score'].transform('max')
    df = df[df['max_v'] == df['score']]
    df = df.groupby('group_id').agg({'player_id':'min'}).reset_index()

    return df[['group_id', 'player_id']]

# ======================================================================
# 1204. Last Person to Fit in the Bus
# ======================================================================

def last_passenger(queue: pd.DataFrame) -> pd.DataFrame:

    queue.sort_values(by='turn', inplace=True)
    queue['tot'] = queue.weight.cumsum()
    name = queue[queue['tot'] <= 1000].tail(1)

    return name[['person_name']]

# ======================================================================
# 1225. Report Contiguous Dates
# ======================================================================

def report_contiguous_dates(failed: pd.DataFrame, succeeded: pd.DataFrame) -> pd.DataFrame:

    failed.rename(columns={'fail_date':'date'}, inplace=True)
    failed['period_state'] = 'failed'

    succeeded.rename(columns={'success_date':'date'}, inplace=True)
    succeeded['period_state'] = 'succeeded'

    df = pd.concat([failed, succeeded], axis=0).sort_values('date')
    df = df[df.date.dt.year == 2019]

    df['bf'] = df['period_state'].shift(1)
    df['id'] = (df.period_state != df.bf).cumsum() ###

    df = df.groupby('id').agg(
        period_state = ('period_state', lambda x: x.iloc[0]), # 1st value
        start_date = ('date', min),
        end_date = ('date', max)
    ).reset_index().sort_values('id')

    return df[['period_state', 'start_date', 'end_date']]

# ======================================================================
# 1270. All People Report to the Given Manager
# ======================================================================

def find_reporting_people(employees: pd.DataFrame) -> pd.DataFrame:

    em = employees

    df = em[em.manager_id == 1]['employee_id'].unique() # [1, 2, 77] array : real boss
    df = em[em.manager_id.isin(df)]['employee_id'].unique() # [1, 2, 4, 77] : 1 manager
    df = em[em.manager_id.isin(df)]['employee_id'].unique() # 2 managers
    df = em[em.manager_id.isin(df)]['employee_id'].unique() # [1, 2, 4, 7, 77] : 3 managers
    df = df[df != 1]

    return pd.DataFrame({'employee_id':df})

# ======================================================================
# 1285. Find the Start and End Number of Continuous Ranges
# ======================================================================

def find_continuous_ranges(logs: pd.DataFrame) -> pd.DataFrame:

    logs['diff1'] = logs.log_id.diff().fillna(0)
    logs['diff2'] = logs.log_id.diff(-1).fillna(0)

    start = logs[logs.diff1 != 1][['log_id']].reset_index(drop=True).rename(columns={'log_id':'start_id'})
    end   = logs[logs.diff2 != -1][['log_id']].reset_index(drop=True).rename(columns={'log_id':'end_id'})

    return pd.concat([start, end], axis=1)

# ======================================================================
# 1303. Find the Team Size
# ======================================================================

def team_size(employee: pd.DataFrame) -> pd.DataFrame:

    team = employee['team_id'].value_counts() # series
    # This uses the pandas map function to map each team_id
    # to a corresponding value in the teams dictionary or series
    employee['team_size'] = employee['team_id'].map(team)
    return employee[['employee_id', 'team_size']]

# ======================================================================
# 1308. Running Total for Different Genders
# ======================================================================

def running_total(scores: pd.DataFrame) -> pd.DataFrame:

    df = scores.sort_values(by=['gender', 'day'])
    df['total'] = df.groupby('gender')['score_points'].cumsum()
    return df[['gender', 'day', 'total']]

# ======================================================================
# 1321. Restaurant Growth
# ======================================================================

def restaurant_growth(customer: pd.DataFrame) -> pd.DataFrame:
    
    df = customer.groupby('visited_on')['amount'].sum() # no reset)index()
    df = df.rolling(7).agg(['sum', 'mean']).round(2).reset_index().dropna()
    return df.rename(columns={'sum':'amount', 'mean':'average_amount'})

# ======================================================================
# 1322. Ads Performance
# ======================================================================

def ads_performance(ads: pd.DataFrame) -> pd.DataFrame:

    ctr = ads.groupby('ad_id')['action'].apply(
        lambda x: round(
            sum(x == 'Clicked') / (sum(x == 'Clicked') + sum(x == 'Viewed')) * 100
            if (sum(x == 'Clicked') + sum(x == 'Viewed')) > 0 else 0.00
            , 2
        )
    ).reset_index()

    ctr.columns = ['ad_id', 'ctr']
    ctr.sort_values(by=['ctr', 'ad_id'], ascending=[False, True], inplace=True)

    return ctr

# ======================================================================
# 1336. Number of Transactions per Visit
# ======================================================================

def draw_chart(visits: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:

    df = transactions.groupby(['user_id', 'transaction_date']).size().reset_index(name='transactions_count')
    df = visits.merge(df, left_on=['user_id', 'visit_date'], right_on=['user_id', 'transaction_date'], how='left').fillna(0)
    df = df.groupby('transactions_count', as_index=False).agg(visits_count=('user_id', 'count'))
    idx = pd.DataFrame({'transactions_count':range(int(max(df.transactions_count))+1)})
    return idx.merge(df, on='transactions_count', how='left').fillna(0)

# ======================================================================
# 1355. Activity Participants
# ======================================================================

def activity_participants(friends: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:

    df = friends.groupby('activity').size().reset_index(name='cnt')
    df = df[(df['cnt'] != df['cnt'].max()) & (df['cnt'] != df['cnt'].min())]
    return df[['activity']]

# ======================================================================
# 1369. Get the Second Most Recent Activity
# ======================================================================

def second_most_recent(user_activity: pd.DataFrame) -> pd.DataFrame:

    df = user_activity.sort_values('endDate')
    df = df.groupby('username').tail(2)

    return df.groupby('username').head(1)

# ======================================================================
# 1384. Total Sales Amount by Year
# ======================================================================

from datetime import datetime

def total_sales(product: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:

    df = pd.DataFrame({
        'report_year': ['2018', '2019', '2020'],
        's_year' : [datetime(2018,1,1), datetime(2019,1,1), datetime(2020,1,1)],
        'e_year' : [datetime(2018,12,31), datetime(2019,12,31), datetime(2020,12,31)]
    })

    df = sales.merge(df, how='cross')
    df['stt'] = df[['s_year', 'period_start']].max(axis=1) # get the larger one
    df['end'] = df[['e_year', 'period_end']].min(axis=1) # get the smaller one

    df['days'] = (df['end'] - df['stt']).dt.days + 1
    df['total_amount'] = df['average_daily_sales'] * df['days']

    df = df[df['days'] > 0]
    df = df.merge(product, on='product_id', how='left')

    return df[['product_id', 'product_name', 'report_year', 'total_amount']]

# ======================================================================
# 1435. Create a Session Bar Chart
# ======================================================================

def create_bar_chart(sessions: pd.DataFrame) -> pd.DataFrame:

    labels = ['[0-5>','[5-10>','[10-15>','15 or more']
    bins = [0, 5*60, 10*60, 15*60, float('inf')]

    df = pd.cut(sessions.duration, labels=labels, bins=bins) ###
    df = df.value_counts().reset_index(name='total')

    return df.rename(columns={'duration':'bin'})

# ======================================================================
# 1440. Evaluate Boolean Expression ###
# ======================================================================

def eval_expression(variables: pd.DataFrame, expressions: pd.DataFrame) -> pd.DataFrame:
    
    df = expressions.merge(variables, left_on='left_operand', right_on='name', how='left')
    df = df.merge(variables, left_on='right_operand', right_on='name', how='left')

    df = df.assign(
        value = np.where(df.operator == '<', df.value_x < df.value_y,
                np.where(df.operator == '>', df.value_x > df.value_y, df.value_x == df.value_y))
    )

    df['value'] = df['value'].astype(str).replace({'True':'true', 'False':'false'})

    return df[['left_operand', 'operator', 'right_operand', 'value']]

# ======================================================================
# 1479. Sales by Day of the Week ###
# ======================================================================

def sales_by_day(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:

    df = items.merge(orders, on='item_id', how='left').rename(columns={'item_category':'category'})
    week = pd.CategoricalDtype(
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True # treat it as ordered
    )

    df['dow'] = df['order_date'].dt.day_name().str.capitalize().astype(week)
    df = df.pivot_table(index='category', columns='dow', values='quantity', aggfunc=sum).reset_index()
    return df

# ======================================================================
# 1484. Group Sold Products By The Date
# ======================================================================

def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:

    df = activities.groupby('sell_date').agg(
        num_sold = ('product', 'nunique'),
        products = ('product', lambda x: ','.join(sorted(set(x)))) ###
    ).reset_index()

    df.sort_values(by='sell_date', inplace=True)
    return df
    
# ======================================================================
# 1517. Find Users With Valid E-Mails ###
# ======================================================================

def valid_emails(users: pd.DataFrame) -> pd.DataFrame:

    # * : those can be zero or more
    # $ : ending

    valid = r"[a-zA-Z][a-zA-Z0-9._-]*\@leetcode\.com$"
    return users[users.mail.str.match(valid)]

# ======================================================================
# 1527. Patients With a Condition ###
# ======================================================================

def find_patients(patients: pd.DataFrame) -> pd.DataFrame:

    # "DIAB1" is matched only when it's not part of a larger word 
    return patients[patients['conditions'].str.contains(r'\bDIAB1')]

# ======================================================================
# 1543. Fix Product Name Format
# ======================================================================

def fix_name_format(sales: pd.DataFrame) -> pd.DataFrame:

    sales['product_name'] = sales['product_name'].str.lower().str.strip() ###
    sales['sale_date'] = sales['sale_date'].dt.strftime('%Y-%m')

    return sales.groupby(['product_name', 'sale_date']).size().reset_index(name='total')

# ======================================================================
# 1613. Find the Missing IDs
# ======================================================================

def find_missing_ids(customers: pd.DataFrame) -> pd.DataFrame:

    ids = list(range(1, max(customers['customer_id']+1)))
    df = pd.DataFrame({'ids':ids})

    return df[~df.ids.isin(customers.customer_id)]

# ======================================================================
# 1635. Hopper Company Queries I
# ======================================================================

def hopper_company(drivers: pd.DataFrame, rides: pd.DataFrame, accepted_rides: pd.DataFrame) -> pd.DataFrame:
    
    dr = drivers[drivers.join_date.dt.year <= 2020]
    dr['month'] = dr['join_date'].dt.month
    dr.loc[dr['join_date'].dt.year < 2020, 'month'] = 1
    dr = dr['month'].value_counts().reset_index()

    df = pd.DataFrame({'month':np.arange(1,13)})
    df = df.merge(dr, on='month', how='left').fillna(0)
    df['active_drivers'] = df['count'].cumsum()

    ar = accepted_rides.merge(rides, how='left', on='ride_id')
    ar = ar[ar['requested_at'].dt.year == 2020]
    ar['month'] = ar['requested_at'].dt.month
    ar = ar['month'].value_counts().reset_index()
    ar.rename(columns={'count':'accepted_rides'}, inplace=True)

    df = df.merge(ar, on='month', how='left').fillna(0)
    return df[['month', 'active_drivers', 'accepted_rides']]

# ======================================================================
# 1667. Fix Names in a Table
# ======================================================================

def fix_names(users: pd.DataFrame) -> pd.DataFrame:

    users['name'] = users['name'].str.lower().str.capitalize()
    return users.sort_values('user_id')

# ======================================================================
# 1683. Invalid Tweets
# ======================================================================

def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    return tweets[tweets['content'].str.len() > 15][['tweet_id']]

# ======================================================================
# 1699. Number of Calls Between Two Persons
# ======================================================================

def number_of_calls(calls: pd.DataFrame) -> pd.DataFrame:

    df1 = calls.rename(columns={'from_id':'person1', 'to_id':'person2'})
    df2 = calls.rename(columns={'from_id':'person2', 'to_id':'person1'})

    df = pd.concat([df1, df2], axis=0)
    df = df[df['person1'] < df['person2']]

    df = df.groupby(['person1', 'person2']).agg(
        call_count=('duration', 'count'),
        total_duration=('duration', 'sum')
    ).reset_index()

    return df

# ======================================================================
# 1767. Find the Subtasks That Did Not Execute
# ======================================================================

def find_subtasks(tasks: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:

    df = tasks['task_id'].repeat(tasks['subtasks_count']).to_frame('task_id')
    df['subtask_id'] = df.groupby('task_id').cumcount() + 1
    df = pd.concat([df, executed])

    return df.drop_duplicates(keep=False)

# ======================================================================
# 1777. Product's Price for Each Store
# ======================================================================

def products_price(products: pd.DataFrame) -> pd.DataFrame:
    
    df = products.pivot(index='product_id', columns='store', values='price').reset_index()
    return df

# ======================================================================
# 1783. Grand Slam Titles ###
# ======================================================================

def grand_slam_titles(players: pd.DataFrame, championships: pd.DataFrame) -> pd.DataFrame:

    # https://pandas.pydata.org/docs/reference/api/pandas.melt.html
    # var_name
    # - Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
    # value_name
    # - Name to use for the ‘value’ column, can’t be an existing column label.
    
    df = pd.melt(championships, id_vars='year', var_name='game', value_name='player_id')
    df = df.groupby('player_id').size().reset_index(name='grand_slams_count')
    df = df.merge(players, on='player_id', how='left')
    return df[['player_id', 'player_name', 'grand_slams_count']]
    
# ======================================================================
# 1795. Rearrange Products Table ###
# ======================================================================

def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:

    df = pd.melt(products, id_vars='product_id', value_vars=['store1', 'store2', 'store3'],
                 var_name='store', value_name='price')
    
    return df.dropna()

# ======================================================================
# 1831. Maximum Transaction Each Day
# ======================================================================

def find_maximum_transaction(transactions: pd.DataFrame) -> pd.DataFrame:

    transactions['dt'] = transactions['day'].dt.strftime('%Y-%m-%d')
    transactions['max'] = transactions.groupby('dt')['amount'].transform('max')

    df = transactions[transactions['max'] == transactions['amount']]

    return df[['transaction_id']].sort_values('transaction_id')

# ======================================================================
# 1843. Suspicious Bank Accounts
# ======================================================================

def suspicious_bank_accounts(accounts: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:

    df = transactions[transactions['type'] == 'Creditor']
    df['mon'] = df['day'].dt.strftime('%Y-%m')

    df = df.groupby(['account_id', 'mon']).agg({'amount':'sum'}).reset_index()
    df = df.merge(accounts, on='account_id', how='left')

    df.sort_values(['account_id', 'mon'], inplace=True)

    df['b_mon'] = (pd.to_datetime(df['mon']) - pd.DateOffset(months=1)).dt.to_period("M") # previous month
    df['prev'] = df.groupby('account_id')['mon'].shift(1)

    df['ex1'] = df['amount'] > df['max_income']
    df['ex2'] = df.groupby('account_id')['ex1'].shift(1)

    df = df[df['ex1'] & df['ex2'] & (df['prev'] == df['b_mon'])]

    return df[['account_id']].drop_duplicates()

# ======================================================================
# 1853. Convert Date Format
# ======================================================================

def convert_date_format(days: pd.DataFrame) -> pd.DataFrame:

    # %d : 09
    # %-d : 9
    days['day'] = days.day.dt.strftime('%A, %B %-d, %Y') # Tuesday, April 12, 2022
    return days

# ======================================================================
# 1873. Calculate Special Bonus
# ======================================================================

def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:

    df = employees.sort_values('employee_id')
    df['bonus'] = np.where((df['employee_id'] % 2 == 1) & (~df['name'].str.startswith('M')), df['salary'], 0)
    
    return df[['employee_id', 'bonus']]    

# ======================================================================
# 1907. Count Salary Categories
# ======================================================================

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:

    low = len(accounts[accounts.income < 20000])
    avg = len(accounts[(accounts.income >= 20000) & (accounts.income <= 50000)])
    hig = len(accounts[accounts.income > 50000])

    return pd.DataFrame({
        'category' : ['Low Salary', 'Average Salary', 'High Salary'],
        'accounts_count' : [low, avg, hig]
    })

# ======================================================================
# 1917. Leetcodify Friends Recommendations
# ======================================================================

def recommend_friends(listens: pd.DataFrame, friendship: pd.DataFrame) -> pd.DataFrame:
    
    df = listens.merge(listens, on =['song_id', 'day'], how='left').rename(columns={'user_id_x':'user1_id', 'user_id_y':'user2_id'})
    df = df[df['user1_id'] != df['user2_id']]
    df = df.groupby(['user1_id', 'user2_id', 'day']).agg(cnt=('song_id','nunique')).reset_index()
    df = df[df.cnt >= 3][['user1_id', 'user2_id']]

    df = df.groupby(['user1_id', 'user2_id']).first().reset_index()

    fr = pd.concat([friendship, friendship.rename(columns={'user1_id':'user2_id','user2_id':'user1_id'})], axis=0)
    fr['fr'] = 'friend'
    df = df.merge(fr, how='left', on=['user1_id', 'user2_id']).rename(columns={'user1_id':'user_id', 'user2_id':'recommended_id'})
    return df[df.fr.isna()][['user_id', 'recommended_id']]
    
# ======================================================================
# 1939. Users That Actively Request Confirmation Messages
# ======================================================================

def find_requesting_users(signups: pd.DataFrame, confirmations: pd.DataFrame) -> pd.DataFrame:

    df = confirmations.sort_values(['user_id', 'time_stamp'])
    df['prev'] = df.groupby('user_id')['time_stamp'].shift(1)
    df['diff'] = (df['time_stamp'] - df['prev']) / pd.Timedelta(hours=1)

    return df[df['diff'] <= 24][['user_id']].drop_duplicates()

# ======================================================================
# 1972. First and Last Call On the Same Day
# ======================================================================

def same_day_calls(calls: pd.DataFrame) -> pd.DataFrame:

    df = pd.concat([calls, calls.rename(columns={'caller_id':'recipient_id', 'recipient_id':'caller_id'})], axis=0)
    df['day'] = df['call_time'].dt.strftime('%Y-%m-%d')

    df['max_t'] = df.groupby(['day', 'caller_id'])['call_time'].transform('max')
    df['min_t'] = df.groupby(['day', 'caller_id'])['call_time'].transform('min')

    max_d = df[df['call_time'] == df['max_t']][['caller_id', 'recipient_id', 'day']]
    min_d = df[df['call_time'] == df['min_t']][['caller_id', 'recipient_id', 'day']]

    df = max_d.merge(min_d, on=['caller_id', 'recipient_id', 'day'], how='inner')
    return df.rename(columns={'caller_id':'user_id'})[['user_id']].drop_duplicates()

# ======================================================================
# 1990. Count the Number of Experiments
# ======================================================================

def count_experiments(experiments: pd.DataFrame) -> pd.DataFrame:

    df = experiments.groupby(['platform', 'experiment_name']).size().reset_index(name='num_experiments')

    pl = pd.DataFrame(['Android', 'IOS', 'Web'], columns=['platform'])
    ex = pd.DataFrame(['Reading', 'Sports', 'Programming'], columns=['experiment_name'])
    co = pl.merge(ex, how='cross')

    df = co.merge(df, on=['platform', 'experiment_name'], how='left').fillna(0)

    return df

# ======================================================================
# 
# ======================================================================

# ======================================================================
# 
# ======================================================================


# ======================================================================
# 
# ======================================================================


# ======================================================================
# 
# ======================================================================

# ======================================================================
# 
# ======================================================================















