
-- =========================================================
-- 176. Second Highest Salary
-- =========================================================

select max(salary) SecondHighestSalary
from employee
where salary != (select max(salary) from employee)

-- =========================================================
-- 180. Consecutive Numbers
-- =========================================================

select distinct num ConsecutiveNums
from (
    select *
        , lag(id,1,0)over(order by id) prev_id
        , lead(id,1,0)over(order by id) next_id
        , lag(num,1,0)over(order by id) prev
        , lead(num,1,0)over(order by id) next
    from logs
) t
where t.num = t.prev
and t.num = t.next
and id - prev_id = 1
and next_id - id = 1

-- =========================================================
-- 184. Department Highest Salary
-- =========================================================
    
with CTE as (
    select d.name Department
        , e.name Employee
        , e.salary
    from Employee e
    left join Department d on e.departmentid = d.id
)
select *
from CTE
where (department, salary) in (
    select department
         , max(salary)
    from cte
    group by 1
)

-- =========================================================
-- 185. Department Top Three Salaries
-- =========================================================
    
with CTE AS (
    select d.name department
        , e.name employee
        , e.salary
        , dense_rank()over(partition by d.name order by salary desc) rk
    from Employee e
    left join Department d on e.departmentid = d.id
)
select department
     , employee
     , salary
from CTE
where rk <= 3

-- =========================================================
-- 196. Delete Duplicate Emails
-- =========================================================

delete p1
from person p1
join person p2
where p1.id > p2.id and p1.email = p2.email

-- =========================================================
-- 197. Rising Temperature
-- =========================================================

-- simple way
select w1.id
from weather w1, weather w2
where DATEDIFF(w1.recordDate, w2.recordDate) = 1
and w1.temperature > w2.temperature 

-- complicated way
select id
from (
    select *
         , lag(recordDate, 1)over(order by recordDate) prev_date
         , lag(temperature, 1)over(order by recordDate) prev_temp
    from weather
) t
where t.prev_temp is not null
and DATEDIFF(t.recordDate, t.prev_date) = 1
and t.temperature - t.prev_temp > 0

-- =========================================================
-- 262. Trips and Users
-- =========================================================
    
select t.request_at Day
     , round(avg(case when status like 'cancelled%' then 1 else 0 end),2) 'Cancellation Rate'
from trips t
inner join users u1 on t.client_id = u1.users_id and u1.banned = 'No'
inner join users u2 on t.driver_id = u2.users_id and u2.banned = 'No'
where t.request_at between '2013-10-01' and '2013-10-03'
group by 1

-- =========================================================
-- 534. Game Play Analysis III
-- =========================================================

select player_id
     , event_date
     , sum(games_played) over(partition by player_id order by event_date) games_played_so_far
from activity
group by 1,2

-- =========================================================
-- 550. Game Play Analysis IV
-- =========================================================

select round(
    count(distinct player_id) / (select count(distinct player_id) from activity), 2
    ) fraction
from activity
where (player_id, DATE_SUB(event_date, interval 1 day))
    in (select player_id, min(event_date) from activity group by 1)

-- =========================================================
-- 569. Median Employee Salary
-- =========================================================

with T1 as (
    select *
        , RANK()over(partition by company order by salary, id) rk
    from employee
), T2 as (
    select company
         , avg(rk) rk
    from T1
    group by 1
)
select t1.id
     , t1.company
     , t1.salary
from T1 
JOIN T2 on t1.company = t2.company
        and abs(t2.rk - t1.rk) < 1

-- =========================================================
-- 570. Managers with at Least 5 Direct Reports
-- =========================================================

select name
from employee
where id in (
    select managerId
    from employee
    group by managerId
    having count(*) >= 5
)

-- =========================================================
-- 571. Find Median Given Frequency of Numbers
-- =========================================================

with t as (
select *
     , sum(frequency) over(order by num) as cumsum
     , sum(frequency) over() / 2 as med
from numbers
)
select round(avg(num), 1) median
from t
where cumsum - frequency <= med
and cumsum >= med

-- =========================================================
-- 574. Winning Candidate
-- =========================================================

with t as (
    select c.id
        , c.name
        , count(*) cnt
    from candidate c
    left join vote v on c.id = v.candidateId
    group by 1
    order by cnt desc
)
select name
from t
limit 1

-- =========================================================
-- 579. Find Cumulative Salary of an Employee
-- =========================================================

select e1.id
     , e1.month
     , e1.salary + IFNULL(e2.salary,0) + IFNULL(e3.salary,0) as salary
from employee e1
left join employee e2 on e1.id = e2.id and e1.month - e2.month = 1
left join employee e3 on e1.id = e3.id and e1.month - e3.month = 2
where (e1.id, e1.month) not in (select id, max(month) from employee group by 1)
group by 1,2
order by id, month desc

-- =========================================================
-- 585. Investments in 2016
-- =========================================================

select round(sum(tiv_2016),2) tiv_2016
from insurance
where tiv_2015 in (select tiv_2015 from insurance group by tiv_2015 having count(*) > 1)
and (lat, lon) in (select lat, lon from insurance group by lat, lon having count(*) = 1)

-- =========================================================
-- 597. Friend Requests I: Overall Acceptance Rate
-- =========================================================
    
WITH R AS (
    select count(distinct sender_id, send_to_id) cnt
    FROM FriendRequest
), A AS (
    select count(distinct requester_id, accepter_id) cnt
    FROM RequestAccepted
)
select round(IFNULL(a.cnt / r.cnt, 0), 2) accept_rate
from r, a

-- =========================================================
-- 601. Human Traffic of Stadium
-- =========================================================

select distinct a.*
from stadium a, stadium b, stadium c
where a.people >= 100 and b.people >= 100 and c.people >= 100
and (
    (a.id - b.id = 1 and b.id - c.id = 1) or
    (c.id - b.id = 1 and b.id - a.id = 1) or
    (b.id - a.id = 1 and a.id - c.id = 1)
)
order by visit_date

-- =========================================================
-- 602. Friend Requests II: Who Has the Most Friends
-- =========================================================

select id
     , count(*) num
from (
    select requester_id id from RequestAccepted
    union all
    select accepter_id id from RequestAccepted
) t
group by 1
order by num desc
limit 1

-- =========================================================
-- 608. Tree Node
-- =========================================================
    
select id
     , case when p_id is null then 'Root'
            when id in (select distinct p_id from tree) then 'Inner'
            else 'Leaf' end as type
FROM Tree

-- =========================================================
-- 612. Shortest Distance in a Plane
-- =========================================================

select min(round(sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)), 2)) shortest
from point2d p1, point2d p2
where p1.x != p2.x or p1.y != p2.y

-- =========================================================
-- 613. Shortest Distance in a Line
-- =========================================================

# NULLIF(A, B) : returns NULL if A = B

select min(nullif(abs(p1.x - p2.x),0)) shortest
from point p1, point p2

-- =========================================================
-- 615. Average Salary: Departments VS Company
-- =========================================================
    
with c as (
    select DATE_FORMAT(pay_date, '%Y-%m') pay_month
         , avg(amount) avg_sal
    from salary s
    group by 1
), t as (
    select DATE_FORMAT(pay_date, '%Y-%m') pay_month
         , e.department_id
         , avg(amount) avg_sal
    from salary s
    left join employee e on s.employee_id = e.employee_id
    group by 1,2
)
select t.pay_month
     , t.department_id
     , case when c.avg_sal = t.avg_sal then 'same'
            when c.avg_sal > t.avg_sal then 'lower'
            else 'higher' end as comparison
from t
left join c on t.pay_month = c.pay_month

-- =========================================================
-- 618. Students Report By Geography ###
-- =========================================================

select MAX(case when continent = 'America' then name end) as America
     , MAX(case when continent = 'Asia' then name end) as Asia
     , MAX(case when continent = 'Europe' then name end) as Europe
from (select *
           , row_number()over(partition by continent order by name) rk
      from student) t
group by rk

-- =========================================================
-- 626. Exchange Seats
-- =========================================================

select case when id = (select max(id) from seat) and id % 2 = 1 then id
            when id % 2 = 1 then id + 1
            else id - 1 end as id
     , student
from seat
order by id

-- =========================================================
-- 627. Swap Salary ###
-- =========================================================

UPDATE salary
SET sex = case when sex='m' then 'f' else 'm' end;

-- =========================================================
-- 1083. Sales Analysis II ###
-- =========================================================

select s.buyer_id
from sales s
left join product p on s.product_id = p.product_id
group by 1
having sum(case when product_name = 'S8' then 1 else 0 end) > 0
and sum(case when product_name = 'iPhone' then 1 else 0 end) = 0

-- =========================================================
-- 1084. Sales Analysis III
-- =========================================================
    
select distinct s.product_id
     , p.product_name
from Sales s
left join product p on s.product_id = p.product_id
group by p.product_id
having min(sale_date)  >= '2019-01-01'
and max(sale_date) <= '2019-03-31'

-- =========================================================
-- 1097. Game Play Analysis V ###
-- =========================================================
    
select install_dt
     , count(distinct player_id) installs
     , round(
        sum(event_date = date_add(install_dt, interval 1 day)) / count(distinct player_id), 2
     ) Day1_retention
from (
    select *
         , min(event_date)over(partition by player_id order by event_date) install_dt
    from activity
) t
group by 1

-- =========================================================
-- 1107. New Users Daily Count
-- =========================================================

select login_date
     , count(*) user_count
from (
    select user_id
        , min(activity_date) login_date
    from traffic
    where activity = 'login'
    group by 1
    ) t
where login_date >= DATE_ADD('2019-06-30', interval -90 day)
group by 1

-- =========================================================
-- 1127. User Purchase Platform
-- =========================================================

with t as (
    select distinct spend_date, 'mobile' platform from spending
    union all
    select distinct spend_date, 'desktop' platform from spending
    union all
    select distinct spend_date, 'both' platform from spending
), p as (
    select spend_date
         , user_id
         , case when count(distinct platform) = 1 then platform
                when count(distinct platform) = 2 then 'both' end platform
         , sum(amount) amt
    from spending
    group by 1, 2
)
select t.spend_date
     , t.platform
     , sum(IFNULL(amt, 0)) total_amount
     , count(distinct user_id) total_users
from t
left join p on t.spend_date = p.spend_date and t.platform = p.platform
group by 1,2

-- =========================================================
-- 1204. Last Person to Fit in the Bus
-- =========================================================

with t as (
    select person_name
         , sum(weight)over(order by turn) tot
    from queue
)
select person_name
from t
where t.tot = (select max(tot) from t where t.tot <= 1000)

-- =========================================================
-- 1212. Team Scores in Football Tournament
-- ========================================================= 
    
select team_id
     , team_name
     , sum(case when t.team_id = m.host_team and host_goals > guest_goals then 3
                when t.team_id = m.guest_team and guest_goals > host_goals then 3
                when host_goals = guest_goals then 1
                else 0 end) num_points
from teams t
left join matches m on t.team_id = m.host_team or t.team_id = m.guest_team
group by team_id
order by num_points desc, team_id

-- =========================================================
-- 1225. Report Contiguous Dates ###
-- ========================================================= 
    
with t as (
    select fail_date dt, 'failed' period_state from failed
    union all
    select success_date dt, 'succeeded' period_state from succeeded
)
select period_state
     , min(dt) start_date
     , max(dt) end_date
from (
    select *
         , rank()over(partition by period_state order by dt) rk
    from t
    where year(t.dt) = 2019
) t
group by period_state, date_sub(dt, interval rk day)
order by dt

-- =========================================================
-- 1285. Find the Start and End Number of Continuous Ranges
-- ========================================================= 

select min(log_id) start_id
     , max(log_id) end_id
from (
    select *
         , row_number()over(order by log_id) rk
    from logs
) t
group by log_id - rk

-- =========================================================
-- 1308. Running Total for Different Genders
-- ========================================================= 

select gender
     , day
     , sum(score_points)over(partition by gender order by day) total
from scores
group by 1,2

-- =========================================================
-- 1321. Restaurant Growth 
-- =========================================================
    
select visited_on
     , amount
     , round(amount / 7, 2) average_amount
from (
    select visited_on
        , sum(amount)over(order by visited_on rows between 6 preceding and current row) amount ###
    from (select visited_on, sum(amount) amount from customer group by 1) t
    group by 1
) t
where visited_on >= (select DATE_ADD(min(visited_on), interval 6 day) from customer)

# ROWS BETWEEN 6 PRECEDING AND CURRENT ROW 
# - the window frame, including the current row and the 6 preceding rows

-- =========================================================
-- 1322. Ads Performance
-- ========================================================= 

select ad_id
     , round(IFNULL(
         sum(action = 'Clicked') / (sum(action = 'Clicked') + sum(action = 'Viewed')) * 100, 0
         ), 2) ctr
from Ads
group by ad_id
order by ctr desc, ad_id

-- =========================================================
-- 1336. Number of Transactions per Visit ###
-- =========================================================     

-- the RECURSIVE keyword should be placed immediately after WITH
with RECURSIVE cte as (
    select v.user_id
         , v.visit_date
         , count(t.user_id) cnt
    from visits v
    left join transactions t on v.user_id = t.user_id and v.visit_date = t.transaction_date
    group by 1,2
), n as (
    select 0 as num
    union all
    select num + 1 from n where num < (select max(cnt) from CTE)
)
select n.num transactions_count
     , count(cte.user_id) visits_count
from n
left join cte on n.num = cte.cnt
group by 1

-- =========================================================
-- 1369. Get the Second Most Recent Activity
-- =========================================================     

select username
     , activity
     , startDate
     , endDate
from (
    select *
         , dense_rank()over(partition by username order by endDate desc) rk
         , count(*)over(partition by username) cnt
    from useractivity
) t
where t.rk = 2 or t.cnt = 1

-- =========================================================
-- 1384. Total Sales Amount by Year
-- =========================================================     

with RECURSIVE T AS (
    select min(period_start) days
    from sales
    union all 
    select DATE_ADD(days, interval 1 day)
    from t
    where days <= (select max(period_end) from sales)
)
select s.product_id
     , p.product_name
     , LEFT(t.days,4) report_year
     , sum(s.average_daily_sales) total_amount
from sales s
left join product p on s.product_id = p.product_id
left join t on t.days between s.period_start and s.period_end
group by 1,2,3
order by 1,3

-- =========================================================
-- 1393. Capital Gain/Loss
-- =========================================================     

select stock_name
     , sum(case when operation = 'Buy' then -price else price end) capital_gain_loss
from stocks
group by 1

-- =========================================================
-- 1479. Sales by Day of the Week
-- =========================================================     

select i.item_category category
     , sum(case when weekday(order_date) = 0 then quantity else 0 end) Monday
     , sum(case when weekday(order_date) = 1 then quantity else 0 end) Tuesday
     , sum(case when weekday(order_date) = 2 then quantity else 0 end) Wednesday
     , sum(case when weekday(order_date) = 3 then quantity else 0 end) Thursday
     , sum(case when weekday(order_date) = 4 then quantity else 0 end) Friday
     , sum(case when weekday(order_date) = 5 then quantity else 0 end) Saturday
     , sum(case when weekday(order_date) = 6 then quantity else 0 end) Sunday
from items i
left join orders o on o.item_id = i.item_id
group by item_category
order by item_category

-- =========================================================
-- 1484. Group Sold Products By The Date ###
-- ========================================================= 

select sell_date
     , count(distinct product) num_sold
     , group_concat(distinct product order by product separator ',') products
from activities
group by 1
order by 1

-- =========================================================
-- 1484. Group Sold Products By The Date ###
-- ========================================================= 

select sell_date
     , count(distinct product) num_sold
     , group_concat(distinct product order by product separator ',') products
from activities
group by 1
order by 1

-- =========================================================
-- 1511. Customer Order Frequency
-- ========================================================= 

select c.customer_id
     , c.name
from orders o
join customers c on c.customer_id = o.customer_id
join product p on p.product_id = o.product_id
where YEAR(o.order_date) = 2020
group by customer_id
having (
    sum(IF(MONTH(o.order_date) = 6, quantity, 0) * price) >= 100
    AND
    sum(IF(MONTH(o.order_date) = 7, quantity, 0) * price) >= 100
)

-- =========================================================
-- 1517. Find Users With Valid E-Mails
-- ========================================================= 

select *
from users
where mail REGEXP "^[a-zA-Z][a-zA-Z0-9._-]*\\@leetcode\\.com$"

-- =========================================================
-- 1613. Find the Missing IDs
-- ========================================================= 

WITH RECURSIVE num as (
    select 1 ids
    union all
    select ids+1 from num where ids < (select max(customer_id) from customers)
)
select ids
from num
where ids not in (select customer_id from customers)

-- =========================================================
-- 1635. Hopper Company Queries I
-- ========================================================= 

with RECURSIVE m as (
    select 1 mon
    union all
    select mon+1 from m where mon < 12
), d as (
    select driver_id
        , CASE WHEN DATE_FORMAT(join_date, '%Y-%m') <= '2020-01' then '1'
                else MONTH(join_date) end as mon
    from drivers
    where join_date <= '2020-12-31'
), ad as (
    select m.mon
         , count(d.driver_id) d_cnt
    from m
    left join d on d.mon <= m.mon ### cumsum
    group by m.mon
), ac as (
    select ride_id
         , requested_at
         , MONTH(requested_at) mon
    from rides
    join acceptedrides using(ride_id)
    having year(requested_at) = 2020
)
select m.mon month
     , d_cnt active_drivers
     , count(ac.ride_id) accepted_rides
from m
left join ad on m.mon = ad.mon
left join ac on ad.mon = ac.mon
group by ad.mon
order by ad.mon

-- =========================================================
-- 1645. Hopper Company Queries II
-- ========================================================= 

WITH RECURSIVE m as (
    select 1 month
    union all
    select month +1 from m where month < 12
), d as (
    select *
         , case when YEAR(join_date) = 2020 then MONTH(join_date)
                else 1 end month
    from drivers
    where YEAR(join_date) < 2021
), ad as (
    select m.month
         , count(d.driver_id) cnt
    from m
    left join d on d.month <= m.month ### cumsum
    group by 1
), ac as (
    select distinct a.driver_id
         , month(r.requested_at) month
    from acceptedrides a
    left join rides r on r.ride_id = a.ride_id
    where YEAR(r.requested_at) = 2020
)
select ad.month
     , case when ad.cnt = 0 then 0.00
            else round(count(ac.driver_id) / ad.cnt * 100, 2)
            end as working_percentage
from ad
left join ac on ad.month = ac.month
group by ad.month
order by ad.month

-- =========================================================
-- 1667. Fix Names in a Table
-- ========================================================= 

select user_id
     , concat(upper(left(name, 1)), lower(substring(name, 2))) name
from users
order by user_id

-- =========================================================
-- 1683. Invalid Tweets
-- ========================================================= 

select tweet_id
from tweets
where CHAR_LENGTH(content) > 15

-- =========================================================
-- 1699. Number of Calls Between Two Persons
-- ========================================================= 

select LEAST(from_id, to_id) person1
     , GREATEST(from_id, to_id) person2
     , count(*) call_count
     , sum(duration) total_duration
from calls
group by 1,2

-- =========================================================
-- 1709. Biggest Window Between Visits
-- ========================================================= 

select user_id
     , max(diff) biggest_window
from (
    select user_id
         , DATEDIFF(lead(visit_date, 1, '2021-1-1')over(partition by user_id order by visit_date), visit_date) diff
    from uservisits
) t
group by 1

-- =========================================================
-- 1767. Find the Subtasks That Did Not Execute
-- ========================================================= 

with recursive t as (
    select task_id
         , subtasks_count
    from tasks
    union all
    select task_id
         , subtasks_count - 1
    from t
    where subtasks_count > 1
)
select task_id
     , subtasks_count subtask_id
from t
where (task_id, subtasks_count) not in (select * from executed)

-- =========================================================
-- 1831. Maximum Transaction Each Day
-- ========================================================= 

select transaction_id
from transactions
where (DATE(day), amount) in (
    select DATE(day)
        , max(amount) amt
    from transactions
    group by 1
)
order by transaction_id

-- =========================================================
-- 1853. Convert Date Format
-- ========================================================= 

select DATE_FORMAT(day, '%W, %M %e, %Y') day -- Tuesday, April 12, 2022
from days

-- =========================================================
-- 1917. Leetcodify Friends Recommendations
-- ========================================================= 

WITH T AS (
    select l1.user_id user1_id
         , l2.user_id user2_id
         , l1.day
    from listens l1
    join listens l2 on l1.song_id = l2.song_id and l1.day = l2.day and l1.user_id < l2.user_id
    group by 1,2,3
    having count(distinct l1.song_id) >= 3
), T2 as (
    select t.*
    from t
    left join friendship f on t.user1_id = f.user1_id and t.user2_id = f.user2_id
    where f.user1_id is null
)
select user1_id user_id
     , user2_id recommended_id
from t2
union
select user2_id user_id
     , user1_id recommended_id
from t2

-- =========================================================
-- 1939. Users That Actively Request Confirmation Messages
-- ========================================================= 

select distinct user_id
from (
    select *
         , TIMESTAMPDIFF(SECOND, lag(time_stamp)over(partition by user_id order by time_stamp), time_stamp)/3600 diff
    from confirmations
) t
where t.diff <= 24

-- =========================================================
-- 1951. All the Pairs With the Maximum Number of Common Followers
-- ========================================================= 

with t as (
    select r1.user_id user1_id
         , r2.user_id user2_id
         , count(*) cnt
    from relations r1
    join relations r2 on r1.follower_id = r2.follower_id and r1.user_id < r2.user_id
    group by 1,2
)
select user1_id
     , user2_id
from t
where t.cnt = (select max(cnt) from t)

-- =========================================================
-- 1972. First and Last Call On the Same Day
-- ========================================================= 

with t as (
    select caller_id as id1
        , recipient_id as id2
        , call_time
    from calls
    union 
    select recipient_id as id1
        , caller_id as id2
        , call_time
    from calls
), t2 as (
    select distinct id1
         , first_value(id2)over(partition by id1, date(call_time) order by call_time) fst
         , first_value(id2)over(partition by id1, date(call_time) order by call_time desc) lst
    from t
)
select distinct id1 user_id
from t2
where fst = lst

-- =========================================================
-- 1990. Count the Number of Experiments
-- ========================================================= 

with pl as (
    select 'IOS' as platform
    union all
    select 'Android' as platform
    union all
    select 'Web' as platform
), ex as (
    select 'Programming' as experiment_name
    union all
    select 'Reading' as experiment_name
    union all
    select 'Sports' as experiment_name
), co as (
    select *
    from pl
    cross join ex
)
select co.platform
     , co.experiment_name
     , count(e.experiment_id) num_experiments
from co
left join experiments e on co.platform = e.platform and co.experiment_name = e.experiment_name
group by 1, 2
order by 1, 2

-- =========================================================
-- 2004. The Number of Seniors and Juniors to Join the Company
-- ========================================================= 

with t1 as (
    select experience  
         , salary
         , sum(salary)over(partition by experience order by salary) cum
    from candidates
    order by experience, salary
), t2 as (
    select 'Senior' experience
        , IFNULL(count(*), 0) accepted_candidates
        , IFNULL(max(cum), 0) cum
    from t1
    where experience = 'Senior' and cum <= 70000
), t3 as (
    select 'Junior' experience
         , IFNULL(count(*), 0) accepted_candidates
         , IFNULL(max(cum), 0) cum
    from t1
    where experience = 'Junior' and cum <= (select 70000 - cum from t2)
)
select experience
     , accepted_candidates
from t2
union all
select experience
     , accepted_candidates
from t3

-- =========================================================
-- 2066. Account Balance
-- ========================================================= 

select account_id
     , day
     , sum(balance)over(partition by account_id order by day) balance
from (
    select *
         , case when type= 'Deposit' then amount else -amount end balance
    from transactions
) t
group by 1,2
order by account_id, day

-- =========================================================
-- 2118. Build the Equation ###
-- ========================================================= 
    
WITH T AS (
    select factor
        , power
        , IF(factor > 0, '+', '') sign
        , case when power = 0 then ''
                when power = 1 then 'X'
                else CONCAT('X^', power) end power2
    FROM TERMS
)
select CONCAT(GROUP_CONCAT(
    CONCAT(sign, factor, power2) ORDER BY power desc separator ""),
    '=0') equation
from t
    
-- =========================================================
-- 2173. Longest Winning Streak ###
-- ========================================================= 

WITH T1 as (
    select *
        , rank()over(partition by player_id order by match_day) rk1
        , rank()over(partition by player_id, result order by match_day) rk2
    from matches
    order by player_id, match_day
), T2 as (
    select player_id
        , (rk1 - rk2) rk
        , count(*) cnt
    from T1
    where result = 'Win'
    group by 1,2
), T3 as (
    select player_id
         , max(cnt) cnt
    from T2
    group by 1
)
select distinct m.player_id
     , IFNULL(t3.cnt, 0) longest_streak
from matches m 
left join t3 on m.player_id = t3.player_id

-- =========================================================
-- 2199. Finding the Topic of Each Post ###
-- ========================================================= 

select p.post_id
     , IFNULL(GROUP_CONCAT(DISTINCT k.topic_id order by k.topic_id), 'Ambiguous!') topic
from posts p
left join keywords k on CONCAT(' ', LOWER(p.content), ' ') like CONCAT('% ', LOWER(k.word), ' %')
group by p.post_id

-- =========================================================
-- 2298. Tasks Count in the Weekend
-- ========================================================= 

# weekday : 0 Mon
select sum(weekday(submit_date) < 5) weekend_cnt
     , sum(weekday(submit_date) >= 5) working_cnt
from tasks

-- =========================================================
-- 2388. Change Null Values in a Table to the Previous Value
-- ========================================================= 

with t1 as (
    select *
         , row_number()over() rk
    from coffeeshop
), t2 as (
    select *
         , SUM(IF(drink is not null, 1, 0)) over(order by rk) rk2
    from t1
)
select id
     , first_value(drink) over(partition by rk2 order by rk) drink
from t2

-- | id | drink             | rk | rk2 |
-- | -- | ----------------- | -- | --- |
-- | 9  | Rum and Coke      | 1  | 1   |
-- | 6  | null              | 2  | 1   |
-- | 7  | null              | 3  | 1   |
-- | 3  | St Germain Spritz | 4  | 2   |
-- | 1  | Orange Margarita  | 5  | 3   |
-- | 2  | null              | 6  | 3   |

-- =========================================================
-- 2394. Employees With Deductions
-- ========================================================= 

select e.employee_id
from employees e
left join (select employee_id
                , SUM(CEILING(TIMESTAMPDIFF(SECOND, in_time, out_time)/60)) diff
           from logs
           group by 1
           ) l on e.employee_id = l.employee_id
where e.needed_hours * 60 > IFNULL(l.diff, 0)

-- =========================================================
-- 2474. Customers With Strictly Increasing Purchases
-- ========================================================= 

with t1 as (
    select customer_id
         , year(order_date) yr
         , sum(price) price
    from orders
    group by 1,2
), t2 as (
    select *
        , lead(yr, 1, 0)over(partition by customer_id order by yr) yr2
        , lead(price, 1, 0)over(partition by customer_id order by yr) price2
    from t1
    order by customer_id, yr
)
select distinct customer_id
from orders 
where customer_id not in (
    select customer_id
    from t2
    where yr2 != 0
    and (yr2 - yr > 1 or price2 - price <= 0)
)

-- =========================================================
-- 2494. Merge Overlapping Events in the Same Hall
-- ========================================================= 

WITH t1 as (
    select *
         , max(end_day)over(partition by hall_id order by start_day) as max_end
    from hallevents
), t2 as (
    select *
         , lag(max_end, 1)over(partition by hall_id order by start_day) as max_end_prev
    from t1
), t3 as (
    select hall_id
         , start_day
         , end_day
         , sum(IF(start_day <= max_end_prev, 0, 1))over(
            partition by hall_id order by start_day
         ) overlap
    from t2
)
select hall_id
     , min(start_day) start_day
     , max(end_day) end_day
from t3
group by hall_id, overlap
order by hall_id, overlap

-- =========================================================
-- 2701. Consecutive Transactions with Increasing Amounts
-- ========================================================= 

with t1 as (
    select t1.customer_id
        , t1.transaction_id
        , t1.transaction_date dt1
        , t2.transaction_date dt2
        , rank()over(partition by t1.customer_id order by t1.transaction_date) rk
    from transactions t1, transactions t2
    where t1.customer_id = t2.customer_id
    and DATEDIFF(t2.transaction_date, t1.transaction_date) = 1
    and t1.amount < t2.amount
), t2 as (
    select *
        , DATE_SUB(dt1, interval rk day) as diff
    from t1
)
select customer_id
     , min(dt1) consecutive_start
     , max(dt2) consecutive_end
from t2
group by 1, diff
having count(*) >= 2
order by 1

-- =========================================================
-- 2752. Customers with Maximum Number of Transactions on Consecutive Days
-- ========================================================= 

WITH T1 as (
    select *
         , rank()over(partition by customer_id order by transaction_date) rk
    from transactions
), t2 as (
    select *
         , DATE_SUB(transaction_date, interval rk day) diff
    from t1
), t3 as (
    select customer_id
         , count(distinct transaction_id) cnt
    from t2
    group by customer_id, diff
), t4 as (
    select *
         , dense_rank()over(order by cnt desc) as rk2
    from t3
)
select customer_id
from t4
where rk2 = 1
order by 1

-- =========================================================
-- 2820. Election Results
-- ========================================================= 

with t as (
    select v1.candidate 
        , sum(v2.cnt) cnt
    from votes v1
    left join (
        select voter
            , 1 / count(*) cnt
        from votes
        group by 1
    ) v2 on v1.voter = v2.voter
    group by 1    
)
select candidate
from t
where t.cnt = (select max(cnt) from t)
order by t.candidate

-- =========================================================
-- 2854. Rolling Average Steps
-- ========================================================= 

with t as (
    select *
        , lag(steps_date, 2) over (partition by user_id order by steps_date) as prev
        , avg(steps_count) over (partition by user_id order by steps_date
                                 rows between 2 preceding and current row) as s_avg
    from steps
)
select user_id
     , steps_date
     , round(s_avg, 2) rolling_average
from t
where DATEDIFF(steps_date, prev) = 2
order by user_id, steps_date

-- =========================================================
-- 2893. Calculate Orders Within Each Interval
-- ========================================================= 

select interval_no
     , sum(order_count) total_orders
from (
    select *
        , case when minute % 6 != 0 then minute div 6 + 1
                else minute div 6 end interval_no
    from orders
) t
group by 1
order by 1

-- =========================================================
-- 2978. Symmetric Coordinates
-- ========================================================= 

WITH T AS (
    select *
         , row_number()over() as rk
    from coordinates
)

select distinct c1.X
     , c1.Y
from t c1, t c2
where (c1.X = c2.Y and c1.Y = c2.X)
and c1.X <= c1.Y
and c1.rk <> c2.rk
group by 1,2
order by c1.X, c1.Y

-- =========================================================
-- 2991. Top Three Wineries
-- ========================================================= 

WITH T AS (
    select country
          , concat(winery, ' (', sum(points), ')') points
          , rank()over(partition by country order by sum(points) desc, winery asc) rk
    from wineries
    group by country, winery
)
select country
     , max(case when rk=1 then points end) top_winery
     , IFNULL(max(case when rk=2 then points end), 'No second winery') second_winery
     , IFNULL(max(case when rk=3 then points end), 'No third winery') third_winery
from t
group by country
order by country

-- =========================================================
-- 2993. Friday Purchases I
-- ========================================================= 

select ceil(day(purchase_date) / 7) as week_of_month
     , purchase_date
     , sum(amount_spend) total_amount
from purchases
where year(purchase_date) = 2023
and month(purchase_date) = 11
and DAYOFWEEK(purchase_date) = 6 # sun = 0
group by 1
order by 1

-- =========================================================
-- 2994. Friday Purchases II ###
-- ========================================================= 

WITH RECURSIVE T AS (
    select '2023-11-01' purchase_date
    union all
    select DATE_ADD(purchase_date, interval 1 day) purchase_date
    from t
    where purchase_date < '2023-11-30'
)
select floor(dayofmonth(t.purchase_date)/7)+1 week_of_month
     , t.purchase_date
     , IFNULL(sum(p.amount_spend), 0) total_amount
from t
left join purchases p on t.purchase_date = p.purchase_date
where dayname(t.purchase_date) = 'Friday'
group by 1,2
order by 1

-- =========================================================
-- 3054. Binary Tree Nodes
-- ========================================================= 

select N
     , case when P is null then 'Root'
            when N in (select p from tree) then 'Inner'
            else 'Leaf' end as Type
from Tree
order by N

-- =========================================================
-- 3055. Top Percentile Fraud
-- ========================================================= 

select policy_id
     , state
     , fraud_score
from (
    select *
         , percent_rank()over(partition by state order by fraud_score desc) rk
    from fraud
) t
where t.rk <= 0.05
order by 2, 3 desc, 1

-- =========================================================
-- 3059. Find All Unique Email Domains
-- ========================================================= 

# substring_index
# if the last number is positive, it returns all to the left
# -1 : return everything after the last occurrence of the @ symbol 
select substring_index(email, '@', -1) email_domain
     , count(*) count
from emails
where email like '%.com'
group by 1
order by 1

-- =========================================================
-- 3060. User Activities within Time Bounds
-- ========================================================= 

select distinct s1.user_id
from sessions s1
inner join sessions s2
    on s1.user_id = s2.user_id
    and s1.session_type = s2.session_type
    and s1.session_start < s2.session_start
    and TIMESTAMPDIFF(HOUR, s1.session_end, s2.session_start) <= 12
order by user_id

-- =========================================================
-- 3061. Calculate Trapping Rain Water ###
-- ========================================================= 

WITH T AS (
    SELECT *
         , MAX(height) OVER(ORDER BY id) m1
         , MAX(height) OVER(ORDER BY id DESC) m2
    FROM heights
)
SELECT SUM(LEAST(m1, m2) - height) total_trapped_water
FROM T
order by id

-- =========================================================
-- 3087. Find Trending Hashtags ###
-- ========================================================= 

select REGEXP_SUBSTR(tweet, '\#[a-zA-Z]+') hashtag
     , count(*) hashtag_count
from tweets
where year(tweet_date) = '2024'
and month(tweet_date) = '2'
group by 1
order by hashtag_count desc, hashtag desc
limit 3
    
-- =========================================================
-- 3089. Find Bursty Behavior ###
-- ========================================================= 

WITH T AS (
    select *
         , count(post_id)over(partition by user_id order by post_date range between interval 6 day preceding and current row) cnt7
         , count(post_id)over(partition by user_id) / 4 as avg_posts
    from posts
    where post_date between '2024-02-01' and '2024-02-28'
)
SELECT user_id
     , max(cnt7) max_7day_posts
     , max(avg_posts) avg_weekly_posts
FROM T
where cnt7 >= 2 * avg_posts
group by user_id
order by user_id

-- =========================================================
-- 3103. Find Trending Hashtags II ###
-- ========================================================= 

WITH RECURSIVE T AS (
    select SUBSTRING_INDEX(SUBSTRING_INDEX(tweet, '#', -1), " ", 1) tag
        --  , LOCATE('#', REVERSE(tweet)) -- last '#' index
         , SUBSTRING(tweet, 1, LENGTH(tweet) - LOCATE('#', REVERSE(tweet))) as rem 
    from Tweets
    UNION ALL
    select SUBSTRING_INDEX(SUBSTRING_INDEX(rem, '#', -1), " ", 1) tag
         , SUBSTRING(rem, 1, LENGTH(rem) - LOCATE('#', REVERSE(rem))) as rem  
    from t 
    where LOCATE ('#', rem) > 0
)
select CONCAT('#', tag) hashtag
     , count(*) count
from t
group by 1
order by 2 desc, 1 desc
limit 3

-- =========================================================
-- 3118. Friday Purchase III 
-- ========================================================= 

SELECT ROUND(DAY(p.purchase_date)/7) week_of_month
     , u.membership
     , IFNULL(SUM(amount_spend), 0) as total_amount
From users u 
LEFT JOIN purchases p on p.user_id = u.user_id
where YEAR(p.purchase_date) = 2023
and MONTH(p.purchase_date) = 11
and WEEKDAY(p.purchase_date) = 4
group by 1, 2
order by 1, 2

-- =========================================================
-- 3124. Find Longest Calls ###
-- ========================================================= 

WITH T AS (
    select t.first_name
         , c.type
         , CONCAT(
            LPAD(FLOOR(c.duration / 3600), 2, '0'), ':',
            LPAD(FLOOR((c.duration % 3600) / 60), 2, '0'), ':',
            LPAD(FLOOR(c.duration % 60), 2, '0')
         ) duration_formatted
         , rank()over(partition by type order by duration desc) rk
    from contacts t
    left join calls c on t.id = c.contact_id
)
select first_name
     , type
     , duration_formatted
from t
where t.rk <= 3
order by type, duration_formatted desc, first_name

-- =========================================================
-- 3126. Server Utilization Time ###
-- ========================================================= 

WITH T AS (
    select *
         , lead(status_time, 1)over(partition by server_id order by status_time, session_status) after
    from servers
), T2 AS (
    select TIMESTAMPDIFF(SECOND, status_time, after) diff
    FROM t
    where session_status='start'
)
select FLOOR(sum(diff) / (24 * 60 * 60)) total_uptime_Days
from t2

-- =========================================================
-- 3140. Consecutive Available Seats II ###
-- ========================================================= 

WITH T1 AS (
    select *
        , seat_id - row_number()over(order by seat_id) diff
    from cinema
    where free = 1
), T2 AS (
    select min(seat_id) first_seat_id   
         , max(seat_id) last_seat_id
         , count(*) consecutive_seats_len
         , rank()over(order by count(*) desc) rk
    from t1
    group by diff
)
select first_seat_id
     , last_seat_id
     , consecutive_seats_len
from t2
where rk = 1
order by 1

-- =========================================================
-- 3150. Invalid Tweets II ###
-- ========================================================= 

select tweet_id
from tweets
where length(content) > 140
or LENGTH(content) - length(replace(content, '#', '')) > 3
or length(content) - length(replace(content, '@', '')) > 3

-- =========================================================
-- 3156. Employee Task Duration and Concurrent Tasks
-- ========================================================= 

WITH T1 AS (
    select *
        , LEAD(start_time, 1)over(partition by employee_id order by start_time) next_t
        , TIMESTAMPDIFF(SECOND, start_time, end_time) diff
    from tasks
    order by employee_id, start_time
), T2 AS (
    select *
         , case when TIMESTAMPDIFF(SECOND, next_t, end_time) > 0 then TIMESTAMPDIFF(SECOND, next_t, end_time)
                else 0 end as overlap
         , case when TIMESTAMPDIFF(SECOND, next_t, end_time) > 0 then 1 else 0 end as cnt
    FROM T1
)
select employee_id
     , FLOOR(SUM(diff - overlap) / 3600) as total_task_hours
     , MAX(cnt) + 1 max_concurrent_tasks
from T2
group by 1

-- =========================================================
-- 3166. Calculate Parking Fees and Duration
-- ========================================================= 

WITH T AS (
    select *
        , TIMESTAMPDIFF(SECOND, entry_time, exit_time) / 3600 hr
    FROM parkingtransactions
), P AS (
    select car_id
         , lot_id
         , SUM(hr) time_sp
    FROM t
    group by 1,2
)
select t.car_id
     , round(sum(fee_paid), 2) total_fee_paid
     , round(sum(fee_paid)/sum(hr), 2) avg_hourly_fee
     , p.lot_id most_time_lot
from t
left join (
    select *
         , rank()over(partition by car_id order by time_sp desc) rk
    FROM p
) p on t.car_id = p.car_id and p.rk = 1
group by t.car_id
order by t.car_id

-- =========================================================
-- 3172. Second Day Verification
-- ========================================================= 

select e.user_id
from emails e
left join texts t on e.email_id = t.email_id and t.signup_action = 'Verified'
where DATEDIFF(t.action_date, e.signup_date) = 1
order by e.user_id

-- =========================================================
-- 3182. Find Top Scoring Students ###
-- ========================================================= 

select s.student_id
from students s
join courses c on s.major = c.major
left join enrollments e on s.student_id = e.student_id
                        and c.course_id = e.course_id
                        and e.grade='A'
group by 1
having count(c.course_id) = sum(e.grade='A')
order by 1

-- =========================================================
-- 
-- ========================================================= 




-- =========================================================
-- 
-- ========================================================= 


-- =========================================================
-- 
-- ========================================================= 











