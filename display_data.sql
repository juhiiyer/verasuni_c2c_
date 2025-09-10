-- Unified view across all categories
SELECT c.name AS college_name,
       'Academic & Learning' AS category,
       a.subcategory,
       a.f1, a.f2, a.f3, a.f4, a.f5
FROM academic_learning a
JOIN colleges c ON a.college_id = c.college_id

UNION ALL

SELECT c.name AS college_name,
       'Campus Life & Student Activities' AS category,
       cl.subcategory,
       cl.f1, cl.f2, cl.f3, cl.f4, cl.f5
FROM campus_life cl
JOIN colleges c ON cl.college_id = c.college_id

UNION ALL

SELECT c.name AS college_name,
       'Facilities & Accommodation' AS category,
       f.subcategory,
       f.f1, f.f2, f.f3, f.f4, f.f5
FROM facilities f
JOIN colleges c ON f.college_id = c.college_id

UNION ALL

SELECT c.name AS college_name,
       'Support Services' AS category,
       ss.subcategory,
       ss.f1, ss.f2, ss.f3, ss.f4, ss.f5
FROM support_services ss
JOIN colleges c ON ss.college_id = c.college_id

UNION ALL

SELECT c.name AS college_name,
       'General Infrastructure' AS category,
       i.subcategory,
       i.f1, i.f2, i.f3, i.f4, i.f5
FROM infrastructure i
JOIN colleges c ON i.college_id = c.college_id

ORDER BY college_name, category, subcategory;
