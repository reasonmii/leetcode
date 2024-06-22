
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
