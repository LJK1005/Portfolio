
SELECT * FROM student;

SELECT deptno, grade, count(*), avg(weight) AS avg_weight, avg(height) AS avg_weight FROM student
GROUP BY deptno, grade;

SELECT grade, count(*), avg(weight) AS avg_weight FROM STUDENT
WHERE weight > 55
GROUP BY grade
HAVING avg(weight) > 60
ORDER BY avg_weight DESC;

SELECT * FROM PROFESSOR;


SELECT * FROM professor WHERE comm IS NOT NULL;



SELECT * FROM student;
SELECT * FROM professor;

SELECT s.name, s.grade, s.profno, p.name FROM student s, professor p
WHERE s.profno = p.profno(+) ORDER BY p.name;

SELECT s.name, s.grade, s.profno, p.name FROM student s
LEFT OUTER JOIN professor p ON s.profno = p.profno ORDER BY p.name;


SELECT s.name, s.grade, s.profno, p.name FROM student s
LEFT OUTER JOIN professor p ON s.profno = p.profno ORDER BY p.name;

SELECT p.name, s.grade, avg(s.weight) FROM student s, professor p WHERE s.profno = p.profno
GROUP BY p.name, s.grade ORDER BY p.name ASC, s.grade desc;

SELECT p.name, s.grade, avg(s.weight) FROM student s INNER JOIN professor p ON s.profno = p.profno
GROUP BY p.name, s.grade ORDER BY p.name ASC, s.grade DESC;

////////////////

SELECT * FROM student;

SELECT studno, name, grade, weight FROM student ORDER BY grade, weight desc;

SELECT grade, avg(weight) FROM student GROUP BY grade ORDER BY grade;

SELECT s.studno, s.name, s.grade, s.weight, t.avg_weight FROM student s,
(SELECT grade, avg(weight) AS avg_weight FROM student GROUP BY grade) t
WHERE s.grade = t.grade ORDER BY grade, weight desc;

SELECT s.studno, s.name, s.grade, s.weight, t.avg_weight FROM student s,
(SELECT grade, avg(weight) AS avg_weight FROM student GROUP BY grade) t
WHERE s.grade = t.grade;

SELECT s.studno, s.name, s.grade, s.weight FROM student s,
	(SELECT grade, avg(weight) AS avg_weight FROM student GROUP BY grade) t
WHERE s.weight > t.avg_weight AND s.grade = t.grade ORDER BY s.grade, s.weight desc;









