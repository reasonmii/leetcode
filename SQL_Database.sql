
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
-- 
-- ========================================================= 










