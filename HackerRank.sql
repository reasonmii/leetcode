with t as (
    select event_id
         , participant_name
         , dense_rank()over(partition by event_id order by score desc) rk
    from (select event_id
                , participant_name
                , max(score) score
          from scoretable
          group by 1,2
          ) t
    order by participant_name
)
select event_id
     , GROUP_CONCAT(case when rk = 1 then participant_name
                    else null end
                    order by participant_name) first
     , GROUP_CONCAT(case when rk = 2 then participant_name
                    else null end
                    order by participant_name) second
     , GROUP_CONCAT(case when rk = 3 then participant_name
                    else null end
                    order by participant_name) third
from t
where t.rk <= 3
group by 1
order by 1
