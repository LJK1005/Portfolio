SELECT * FROM department;

SELECT deptno, dname from department;

select deptno from student;
select distinct deptno from student; // distinct를 붙이면 중복이 제거됨

select deptno, grade from student;
select distinct deptno, grade from student; // 다수 컬럼에 대하여 distinct 적용시 모든 행이 일치해야 중복으로 처리됨

select dname 학과이름, deptno 학과번호 from department; // 컬럼에 별칭 부여, 홑따옴표는 쓰지 않음
select dname as 학과이름, deptno as 학과번호 from department; // as를 쓸수 있음, 가독성이 좋아짐

select name, sal, sal*12+100 from professor; // select 절에서 산술연산이 가능하며, 조회 결과에 적용되어 출력됨, select 절에서 원본 데이터는 절대로 변하지 않음
// sql에서 조회시에는 문제가 되지 않으나 연산 컬럼을 파이썬으로 가져갈 경우 별칭 지정은 필수적임

// select절에 where을 추가하여 조회 조건을 지정할 수 있음
// 비교연산자(=, > 등), 논리연산자(AND, OR 등) 사용 가능

select studno, name, deptno from student where grade=1; // grade(학년)가 1인 데이터 조회

select studno, name, grade, deptno, weight from student
where weight <= 70; // SQL문에서는 각 명령어를 세미콜론(;)으로 구분하므로 파이썬과 달리 줄바꿈, 들여쓰기 등의 사용이 자유로움

select name, studno, grade, weight, deptno from student where grade=1 and weight >= 70; // AND 사용

// 연습문제1
select name, studno from student;

// 연습문제2
select distinct position from professor;

// 연습문제3
select deptno as 부서, dname as 부서명, loc as 위치 from department;

// 연습문제4
select name as 학생이름, (height-110)*0.9 as 표준체중 from student;

// 연습문제5
select studno, name, grade from student where deptno=101;

// 연습문제6
select profno, name, sal from professor where deptno=101;

// 연습문제7
select studno, name, grade, deptno, height from student where height >= 170;

// 연습문제8
select name, studno, grade, weight, deptno from student where grade=1 or weight >= 70;




