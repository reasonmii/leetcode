
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
-- 184. Department Top Three Salaries
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











