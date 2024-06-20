
-- 176. Second Highest Salary
select max(salary) SecondHighestSalary
from employee
where salary != (select max(salary) from employee)

-- 180. Consecutive Numbers

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

-- 1
