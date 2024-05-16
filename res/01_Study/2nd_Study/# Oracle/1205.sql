select studno, name, weight from student
where weight between 50 and 70; // BETWEEN 연산자는 특정 구간의 데이터를 조회 가능 (이상/이하 기준)
select studno, name, weight from student
where weight >= 50 and weight <= 70; // 위와 같은 효과

select name, grade, deptno from student
where deptno in (102, 201); // IN 연산자는 괄호 내에 묶은 데이터 안에 있는 것과 일치하는 것이 있으면 참이 됨
select name, grade, deptno from student
where deptno = 102 or deptno = 201; // 위와 같은 효과

select name, grade, deptno from student where name like '김%'; // LIKE 연산자는 문자열을 특정하여 지정 가능
// %는 해당 자리에에 어떠한 문자열이라도 있음을 의미

// NULL : 미확인 값으로, 아직 결정되지 않은 값을 의미 (결측치)
// 숫자 0이나 공백과는 다른 개념

select name, position, comm from professor; // COMM에 null 값 존재

select name, position, comm from professor where comm is null; // COMM이 null인 데이터 조회
select name, position, comm from professor where comm is not null; // COMM이 null이 아닌 데이터 조회

select name, grade, deptno from student
where depno = 102 and (grade = 4 or grade = 1); // AND는 OR보다 우선되나 괄호로 묶어서 OR가 먼저 연산됨

// 연습문제1
select * from professor where sal between 300 and 400;

// 연습문제2
select profno, name, position, deptno from professor where position in ('조교수', '전임강사');

// 연습문제3
select deptno, dname, loc from department where dname like '%공학%';
// like는 검색 성능이 낮아서 ES(Elastic Search)라는 자연어 검색엔진을 쓰기도 함

// 연습문제4
select studno, name, grade, profno from student where profno is not null;

// 연습문제5
select name, grade, deptno from student where (deptno = 102 and grade = 4) or grade = 1; // 괄호가 없어도 됨 (AND가 OR보다 우선임)



select name, grade, tel from student order by name;
// ORDER BY 절은 WHERE 절 뒤에 옴, 오름차순(ASC)가 디폴트, DESC는 별도 명시 필요

select studno, name, grade, deptno, userid from student
order by deptno asc, grade desc;
// ,를 이용하여 ORDER BY 절에 여러개의 정렬 옵션을 지정할 수 있음. 먼저 명시한 정렬조건부터 순차적으로 실행

select rownum as RNUM, tbl.* from(
    select deptno, dname from department order by deptno asc) tbl;
// ROWNUM : 데이터를 한번에 조회하면 속도가 느려지므로 데이터 부분 조회를 위하여 결과집합(select절의 결과) 내 레코드에 가상 순번을 부여함
// 해당 순번을 기준으로 쇼핑 사이트 상품 조회 화면 등에서 페이지를 분할하여 데이터가 출력됨
// SELECT를 2중으로 사용, 1차 SELECT 절에 별칭을 붙이고 해당 별칭을 2차 SELECT 절에서 사용
// MySQL의 LIMIT 문과 가장 유사한 기능
// 부분조회는 ROWNUM에 대한 WHERE 절을 이용하여 조건을 걸어서 사용
// ROWNUM은 0이 아닌 1부터 시작하기 때문에 조회 조건 사용에 주의가 필요
// 파이썬 슬라이싱이 n이상, m미만의 방식이면 ROWNUM은 n초과, m이하의 방식으로 정의해야 동일한 결과가 됨
// ROWNUM이 특정 값이거나 특정 값 이상/초과인 조건을 걸 경우 논리 오류로 조회가 되지 않으니 1차 SELECT로 ROWNUM을 확정하고 해당 ROWNUM을 2차 SELECT로 조건을 걸어 조회
// ex) ROWNUM = 5를 그냥 쓰면 조회 결과가 1개인데 ROWNUM 5를 찾는 것이므로 정상 조회가 안됨, 이하/미만은 정상 조회 가능

select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc
    )tbl where rownum <= 3
) where RNUM > 0;

// 연습문제1
select name, grade, idnum from student order by grade desc;

// 연습문제2
select name, grade, deptno from student where deptno = 101 order by birthdate;

// 연습문제3
select name, studno, grade from student order by grade asc, name asc;

// 연습문제4
select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc) tbl
) where RNUM = 5;

select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc
    )tbl where rownum <= 5
) where RNUM > 4;


// 함수는 SQLD에서 오라클 전용 함수가 많이 나옴
// SQL에서 함수 input은 컬럼단위


select name, length(name) from student; // length 함수는 해당 문자의 글자 수를 조회

select name, substr(name, 1, 1) from student;
// substr(컬럼, 시작인덱스, 조회할 문자갯수), 오라클은 파이썬과 달리 1부터 count 하는것에 유의

select name, substr(name, length(name)) from student;
// length(name)과 조합하면 마지막 글자 지정 가능, stbstr에서 마지막 파라미터 지정하지 않으면 끝까지 슬라이싱함

select name, substr(name, 2, 1) from student;

select name, replace(name, '이', 'lee') from student;
// replace(문자열, 변경대상문자열, 변경할문자열) : 없으면 그대로 출력


select concat(name, grade) from student;
select name || grade from student;

select concat(concat(concat(name, ' '), grade), '학년') from student;
select name || ' ' || grade || '학년' from student;
// 오라클의 concat 함수는 input이 파라미터가 2개만 가능하므로 concat을 여러번 사용해야 함
// concat() 대신 ||를 사용하여 문자열을 연결함

select trim(name), ltrim(name), rtrim(name) from student;
// trim, ltrim, rtrim : 공백제거, 파이썬의 strip과 동일

select instr(name, '이'), name from student;
// instr(문자열, 찾기대상) : 파이썬의 find. 특정 문자열이 나타나는 위치 조회 없으면 0

select upper(userid), lower(userid) from student;
// upper, lower : 대/소문자 변환



select sysdate from dual;
// 데이터가 필요없는 sysdate만 출력할 경우 가상 테이블 dual을 조회하여 구문을 완성

select to_char(sysdate, 'YYYYMMDDHH24MIDD') from dual;
// TO_CHAR을 사용하면 원하는 형식으로 날짜 조회 가능


select to_char(sysdate+100, 'YYYY-MM-DD'), to_char(sysdate-7, 'YYYY-MM-DD') from dual;
// 날짜에 숫자 연산은 일 기준으로 수행


// 연습문제1
select name, replace(name, substr(name, 2, 1), '*') from student;
select name, substr(name, 1, 1) || '*' || substr(name, 3, 1) from student;

// 연습문제2
select name, replace(idnum, substr(idnum, 7), '*******') from student;
select name, substr(idnum, 1, 6) || '*******' from student;

// 연습문제3
select name, birthdate from student where birthdate >= '1980-01-01';
select name, to_char(birthdate, 'YYYY-MM-DD') from student
where to_char(birthdate, 'YYYY') > 1980;


// 그룹함수 : 테이블의 전체 행, 하나 이상의 컬럼을 기준으로 컬럼 값에 따라 그룹화하여 결과를 출력하는 함수 (갯수, 최대값, 평균 등)

select count(studno) from student where grade = 3;
// grade가 3인 학생 수 조회


select comm from professor where deptno = 101;
select count(comm) from professor where deptno = 101;
// count와 같은 집계 관련 함수는 Null 값을 제외하고 셈

select count(*) from professor where deptno = 101;
// count 함수에 *을 지정하거나 null이 없는 컬럼을 지정하면 null값도 셀수 있음

select max(sal) from professor;
select min(sal) from professor;
select sum(sal) from professor;
select avg(height) from student;
select avg(weight), sum(weight) from student where deptno = 101;
// 집계함수의 사용


// GROUP BY 문 : 특정 컬럼 값을 기준으로 테이블의 전체 행을 그룹별로 나눔
// WHERE 문과 ORDER BY 절 사이에 들어감 (위치 변경 불가)
// GROUP BY 절에는 반드시 컬럼 이름이 들어가야 함
// SELECT 절에서 집계 함수 없이 나열된 컬럼 이름이나 표현식은 GROUP BY 절에 반드시 포함되야 함 (그루핑을 하거나, 집계를 하거나)

select deptno, name from professor order by deptno;
select deptno, name from professor group by deptno; // name 컬럼의 처리를 명시하지 않았으므로 에러 발생
select deptno, count(name) from professor group by deptno; // 집계함수 count 처리를 하여 deptno별 데이터 개수를 조회

select deptno, count(*), count(comm) from professor group by deptno;


select deptno, grade, count(*), avg(weight)
from student
group by deptno, grade;
// 다중 컬럼을 이용한 그룹별 검색
// 2개 이상의 컬럼을 컴마로 묶어서 다중 그루핑이 가능


// HAVING 절 : GROUP BY 절에 대한 조건식
// SELECT 절과 ORDER BY 절에 집계함수를 쓰는 경우 SELECT 절에서 별칭을 지정하고 해당 별칭을 ORDER BY 절에서 사용하여 연산 시간 단축 가능

select grade, count(*), avg(height) avg_height, avg(weight) avg_weight
from student group by grade
order by avg_height desc;

select grade, count(*), avg(height) avg_height, avg(weight) avg_weight
from student group by grade
having count(*) > 4
order by avg_height desc;

select deptno, grade, count(*), max(height), max(weight)
from student group by deptno, grade
having count(*) >= 3
order by deptno;
// WHERE절에서 집계함수는 사용할 수 없음

// 연습문제1
select max(height), min(height) from student where deptno = 101;

// 연습문제2
select deptno, avg(sal), min(sal), max(sal) from professor group by deptno;

// 연습문제3
select deptno, avg(weight) avg_weight, count(studno) from student group by deptno order by avg_weight desc;

// 연습문제4
select deptno, count(profno) from professor group by deptno having count(profno) <= 2;


select name, department.deptno, dname from professor, department;
// JOIN : 두 개 이상의 테이블을 결합하여 필요한 데이터를 조회하는 기능
// 데이터는 가능한 중복을 피하여 저장하는 것이 좋음 (데이터 원자성)
// 교수 데이터와 학과 데이터를 별도로 저장하여 교수의 학과를 문자열로 조회할 때 두 데이터를 조인시켜 데이터를 조회
// JOIN으로 2개 이상의 테이블을 조회할 때 테이블은 from 절에서 컴마로 여러개의 테이블을 조회
// 테이블 별로 동일한 명칭의 컬럼이 있을 경우 충돌을 회피하기 위해 select 절에서 '테이블이름.컬럼이름' 형식으로 컬럼을 명시
// 아무 JOIN 조건 없이 혹은 잘못된 조건으로 다수의 테이블을 조회하면 모든 1:1 조합을 나열하게 됨 (카디션 곱), 해당 현상을 피해야 함
// 호출하는 테이블에도 별칭을 부여할 수 있음

select name, department.deptno, dname from professor, department
where professor.deptno = department.deptno;

select p.name, d.deptno, d.dname from professor p, department d
where p.deptno = d.deptno;
// EQUI JOIN : 조인 대상 테이블에서 공통 칼럼에 대해 '=' 비교를 명시해 같은 값을 같는 행만을 연결하여 결과를 생성하는 조인

select p.name, d.deptno, d.dname 
from professor p, department d
where p.deptno = d.deptno and p.deptno = 101;


select p.name, d.deptno, d.dname from professor p
inner join department d on p.deptno = d.deptno;

select p.name, d.deptno, d.dname from professor p
inner join department d on p.deptno = d.deptno
where p.deptno = 101;
// INNER JOIN : 결과는 EQUI JOIN과 같음, WHERE에서 공통 컬럼에 조건을 걸었던 EQUI JOIN과 달리 추가로 불러오는 테이블을 INNER JOIN으로 변경
// WHERE은 ON으로 변경하여 JOIN 처리
// 조회하고 싶은 데이터를 확인하여 FROM에서 불러오는 테이블과 INNER JOIN으로 불러오는 테이블을 구분
// INNER JOIN에 추가적인 검색 조건이 필요할 경우 WHERE 절 사용(ON에 사용하면 성능이 대폭 감소)


select s.name, p.name from student s, professor p where s.profno = p.profno; // EQUI JOIN
select s.name, p.name from student s inner join professor p on s.profno = p.profno; // INNER JOIN
// OUTER JOIN : 조건에 부합하지 않는 행들도 포함시켜 결합시킴
// LEFT OUTER : JOIN 절에서 명시한 테이블에서 왼쪽의 테이블에 대하여 조건에 부합하지 않는 데이터까지 조회
// RIGHT OUTER : JOIN 절에서 명시한 테이블에서 오른쪽의 테이블에 대하여 조건에 부합하지 않는 데이터까지 조회
// LEFT / RIGHT JOIN은 순서만 바뀌면 결과는 같기 때문에 하나만 지정하여 쓰는게 좋음
// FULL OUTER : JOIN 절에서 명시한 모든 테이블에 대하여 조건에 부합하지 않는 데이터까지 조회, 시스템 성능에 영향을 크게 주어 잘 쓰지 않음


select s.name, p.name from student s left outer join professor p on s.profno = p.profno;
select s.name, p.name from student s right outer join professor p on s.profno = p.profno;
// OUTER JOIN은 다른 테이블에 공통된 데이터가 없더라도 Null 값을 처리하여 지정한 테이블에 대한 모든 데이터 조회 보장

select s.name, p.name from student s left outer join professor p on s.profno = p.profno;
select s.name, p.name from student s, professor p where s.profno = p.profno(+);
// 오라클에서는 EQUI JOIN의 WHERE 구문에 특정 테이블/컬럼에 (+)을 붙여서 OUTER JOIN을 수행할 수 있음 ((+)가 붙은쪽에서 null값을 허용)

select s.name, p.name from student s right outer join professor p on s.profno = p.profno;
select s.name, p.name from student s, professor p where s.profno(+) = p.profno; // (+) 연산자를 이용한 RIGHT OUTER JOIN


// 서브쿼리 (SubQuery)
// 하나의 SQL문의 처리결과를 다른 SQL 명령문에 전달하기 위해 두 개 이상의 SQL문을 괄호로 하나의 SQL문으로 연결한 형태
// 서브쿼리를 포함하는 SQL문을 메인쿼리라고 칭함

// 단일 행 서브쿼리, 다중 행 서브쿼리에 따라 반환하는 값이 달라 비교연산자와 IN 연산자의 사용 여부가 달라짐

select name, position from professor
where position = (
    select position from professor where name = '전은지');
// 단일 결과를 조건으로 하는 서브쿼리

select name, deptno, grade, height from student
where grade = 1 and height > (
    select avg(height) from student);

select name, dname from student s
inner join department d on s.deptno = d.deptno
where s.deptno = (select deptno from student where name = '이광훈');
// JOIN과 함께 사용하는 서브쿼리

select studno, grade, name from student
where profno in (select profno from professor where sal > 300);
// 다중 행 서브쿼리 : 결과값이 몇 개일지 알 수 없는 서브쿼리는 IN을 사용하여 조건 지정


// 연습문제1
select s.studno, s.name, s.deptno, d.dname, d.loc from student s, department d where d.deptno = s.deptno;

// 연습문제2
select s.studno, s.name, s.deptno, d.dname, s.grade from student s, department d where d.deptno = s.deptno and s.deptno = 102;

// 연습문제3
select s.name, s.grade, p.name, p.position from student s, professor p where s.profno = p.profno;

// 연습문제4
select s.name, s.grade, p.name, p.position from student s inner join professor p on s.profno = p.profno;

// 연습문제5
select name, grade from student where grade = (select grade from student where userid = 'jun123');

// 연습문제6
select name, deptno, weight from student where weight < (
    select avg(weight) from student where deptno = 101);

// 연습문제7
select s.name, s.weight, d.dname, p.name from student s, department d, professor p
where s.profno = p.profno and s.deptno = d.deptno and s.weight < (
    select avg(weight) from student where deptno = (
        select deptno from student where name = '이광훈')
        ); // EQUI JOIN

select s.name, s.weight, d.dname, p.name from student s
inner join department d on s.deptno = d.deptno
inner join professor p on s.profno = p.profno
where s.weight < (
    select avg(weight) from student where deptno = (
        select deptno from student where name = '이광훈')
        ); // INNER JOIN

// 연습문제8
select name, grade, height from student
where grade = (select grade from student where studno = 20101)
and height > (select height from student where studno = 20101);

// 연습문제9
select s.studno, d.dname, s.grade, s.name from student s, department d
where s.deptno = d.deptno and d.dname like '%공학%';

select studno, dname, grade, name from student s
inner join department d on s.deptno = d.deptno
where s.deptno in (select deptno from department where dname like '%공학%'); // 서브쿼리 사용















