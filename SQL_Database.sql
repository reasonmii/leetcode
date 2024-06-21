
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








