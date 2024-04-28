// INSERT 절 : 테이블에 데이터를 추가하는 구문
// 숫자는 홑따옴표 사용, 그 외는 홑따옴표 사용
// 데이터를 나열하는 순서는 컬럼에 맞게 (컬럼 명칭과 데이터타입은 desc 테이블이름; 으로 확인 가능)


insert into student values(
    10110, '홍길동', 'khd85', 1, '851011143098',
    TO_DATE('2013-01-01 11-42-30', 'YYYY-MM-DD HH24:MI:SS'),
    '055)-777-7777', 170, 70, 101, 9903);
// 컬럼 명을 명시하지 않아 컬럼의 순서를 지켜야 함    

insert into student(
    studno, name, userid, grade, idnum, birthdate,
    tel, height, weight, deptno, profno
) values (
    10111, '둘리', 'dooly', 2, '8202021334765',
    to_date('2011-10-01 11:42:30', 'YYYY-MM-DD HH24:MI:SS'),
    '055)-777-7777', 170, 70, 101, 9903
);
// 컬럼 명을 명시하면 순서와 상관없이 데이터 입력 가능
select studno, name, grade from student;

desc department;
// DESC의 Null?에 NOT NULL이라고 명시되면 결측치를 허용하지 않음, 없다면 결측치 허용

insert into department (deptno, dname) values(210, '응용과학');
insert into department (deptno, dname, loc) values (211, '영문학과', NULL);
// 컬럼명을 명시할때 컬럼을 빼거나 컬럼을 명시하고 직접 Null로 입력하면 널값 입력 가능

insert into professor(
    profno, name, userid, position, sal, hiredate, deptno
) values (
    9920, '고길동', 'gilldong', '교수', 450,
    to_date('2014-01-01', 'YYYY-MM-DD'), 102
);
// 날짜 데이터 입력시에는 TO_DATE() 함수를 사용하여 'YYYY-MM-DD HH24:MI:SS' 혹은 'YYYY-MM-DD'로 형식을 변환하여 입력

insert into professor(
    profno, name, userid, position, sal, hiredate, comm, deptno
) values (
    9921, '뚬땜이', 'ttun', '교수', 550, sysdate, null, 102
);
// 현재시각은 SYSDATE 사용
select * from professor;

insert into department values (211, '영문학과2', NULL);
// 기본키(Primary Key) : 컬럼별로 중복을 허용하지 않는 옵션(데이터 무결성), 일반적으로 일련번호(학번, 도서번호 등) 필드에 지정
// 기본키 옵션을 걸어서 중복을 허용하지 않는 것을 무결성 제약 조건이라고 부름 (기본키 제약 조건)
// 학번 등의 기본키는 숫자가 순차적으로 늘어나는 경우가 대부분이라 Sequential이라고도 칭하기도 함 (Seq) 혹은 ID


insert into department(deptno, dname, loc)
values(seq_department.nextval, '국문학과', '5호관');
// 자동 증가 일련번호 속성 : create sequence를 이용하여 insert 절에서 자동으로 값이 증가하도록 데이터를 입력할 수 있어 데이터의 고유성을 유지할 수 있음
// create sequence 시퀀스이름 start with 시작값 increment by 증가값;
// 시퀀스 호출은 insert문에서 시퀀스이름.nextval을 사용함

update professor set position='조교수', sal = 250
where profno = 9903;
// update 절 : 데이터 수정, 값을 변경할 컬럼만 지정하여 변경 가능
select name, position, sal from professor where profno = 9903;

delete from student where studno = 20103;
// delete 절 : 데이터 삭제
select studno, name from student where studno = 20103;


update student set name = '뚬땜이';
delete from student;
// update와 delete 절은 where 절을 사용하지 않으면 전체 데이터의 수정이나 삭제가 발생하므로 주의 필요


update student set birthdate = sysdate where studno = 20101;
// 데이터 입력, 수정, 삭제시에도 함수를 활용할 수 있음
// 날짜를 제외하고는 함수를 사용하는 일이 많지 않음
select name, to_char(birthdate, 'YYYY-MM-DD')
from student where studno = 20103;

commit;
rollback;
// 오라클에서 INSERT, UPDATE, DELETE는 사용할 시점에서는 실제로 반영이 되지 않기 때문에 실제 적용을 위해 COMMIT 하거나 ROLLBACK으로 취소가 가능함 (트렌젝션)
// MySQL에서는 기본 설정이 AUTO-COMMIT (트렌젝션X)이기 때문에 한번 잘못된 수정이 되면 되돌리기 어렵다
// COMMIT을 하면 되돌릴 수 없음


// 연습문제1
desc department;
insert into department (deptno, dname, loc) values (seq_department.nextval, '인터넷정보과', '공학관');
select * from department;

// 연습문제2
update department set loc = '5호관' where loc = '공학관';
select * from department;

// 연습문제3
delete department where deptno = 300;
select * from department;

rollback;

create table Product_test (id number not null, product varchar2(255) not null, reg_date date not null, score number not null, positive_negative varchar2(25) not null, review varchar2(500) not null, primary key(id));
drop table Product_test;
create table PRODUCT_TEST (id number not null, product varchar2(255) not null, reg_date date not null, score number not null, positive_negative varchar2(25) not null, review varchar2(1000) not null, primary key(id));

create table mytable (
    id  number not null,
    memo    varchar2(255) not null,
    reg_date    date not null,
    primary key(id)
);
// CREATE문 : 테이블 생성, DDL
// 컬럼 이름과 데이터 타입, Null 허용여부(안쓰면 허용), Primary Key 컬럼을 지정
// Primary Key는 끝에 Primary Key(컬럼이름)을 쓰거나 특정 컬럼 뒤에 Primary Key를 추가
// Primary Key 지정을 위하여 시퀀스도 생성
// Primary Key는 Null을 허용하지 않음


// 텍스트 타입
// varchar2 : 가변 길이, 지정한 길이 내에서 입력 글자 길이에 맞추어 데이터의 크기가 변함
// char : 입력한 글자에 관련없이 고정된 길이로 저장함, 지정 사이즈를 초과할 경우 에러가 나거나 텍스트를 자름 (우편번호, 주민번호 등 고정된 길이에 대하여 사용)
// long : 가변 길이, 긴 문자열 (게시판 본문 등)에 주로 사용

// 숫자형
// number : 뒤의 괄호를 지정하지 않으면 정수, (전체자리, 소수점자리)를 붙여서 실수형 데이터도 입력 가능

// date : 날짜형

drop table mytable;
// 테이블 삭제는 drop으로 가능하며, 트렌젝션이 아니기 때문에 한번 삭제하면 복구할 수 없음.
// alter table : 테이블 수정, 무엇을 변경할지 뒤에 붙임

alter table mytable rename to mytest;
// 테이블 이름 변경

alter table mytest add edit_date date not null;
// 컬럼 추가
// 이미 데이터가 있는 테이블에 컬럼을 추가할 때는 Not Null로는 지정할 수 없음 (기존 데이터에 대해서는 Null이 되야 하므로)
// Not Null로 추가해야 한다면 우선 Null로 추가하여 모든 데이터를 업데이트 하고 컬럼을 Not Null로 변경해야 함
desc mytest;

alter table mytest rename column memo to content;
// 컬럼 이름 변경
desc mytest;

alter table mytest modify(content varchar2(512) not null);
// 컬럼의 데이터타입 변경
// 데이터 손실을 방지하기 위해 컬럼의 저장공간을 줄일 수 없음 (Null만 있거나 행이 없다면 가능)
// Null만 있는 컬럼은 데이터 유형을 변경할 수 있음
// Null이 없을 경우에만 Not Null로 변경 가능

alter table mytest drop column edit_date;
// 컬럼 삭제

alter table mytest drop primary key;
// 기본키 속성 해제

alter table mytest add constraint pk_mytest_id primary key(id);
// 기본키 속성 설정
// 컬럼 수정으로 기본키 속성을 부여할 경우 속성 이름을 지정해야함 (Create에서는 자동 지정함)


// 테이블의 참조관계
// 어떤 데이터가 다른 테이블의 컬럼에 있는 데이터여야 할 경우 (교수 테이블의 학과번호는 학과 테이블에 있어야 함) 해당 테이블을 참조하여야 함 (참조 무결성 제약 조건)
// 특정 컬럼이 다른 테이블의 기본키를 참조하는 경우가 가장 많음 (참조키, 외래키, Foreign Key로 칭함)
// Primary Key 끼리 참조가 되는 경우 1:1 관계이며, Primary Key - 일반 칼럼은 1:n의 관계 (카디널리티)
// 데이터 정규화 : 데이터의 효율성을 위하여 참조키를 이용해 데이터를 쪼개놓은 형태 (ex. 학과명(+a)을 각 학생에게 일일히 부여하지 않고 학과 번호만 부여하여 효율성을 늘림)
// Primary Key 끼리 1:1 참조는 사실상 데이터를 합칠수 있음 (역정규화)

create sequence seq_grade start with 1 increment by 1;

create table grade(
    id  number not null,
    studno number not null,
    subject varchar2(100) not null,
    point number not null,
    reg_date date not null,
    primary key(id),
    foreign key (studno) references student(studno)
);
// 참조키 설정
// 참조를 받는 데이터는 참조하는 데이터가 모두 삭제되기 전까지는 삭제가 불가능 (ex. 학과 내 학생을 두고 학과를 삭제하는것은 불가)

insert into grade (id, studno, subject, point, reg_date)
values (seq_grade.nextval, 1234, '데이터베이스', 98, sysdate);
// 참조 대상에 데이터가 없을 경우 입력 시도시 에러 발생

insert into grade(id, studno, subject, point, reg_date)
values (seq_grade.nextval, 10101, '데이터베이스', 98, sysdate);
// 참조 대상에 있는 데이터를 사용하여 데이터 추가

delete from student where studno = 10101;
// 참조 대상에 있는 원본 데이터는 참조하는 데이터를 삭제하기 전까지는 삭제할 수 없음
// 기본키 / 참조키에 관한 설명은 자격증 시험에 많이 출제됨
delete from grade where studno = 10101;
delete from student where studno = 10101;
commit;

select * from grade;

// PL/SQL : 오라클에 자체적으로 내장된 언어
// 블록 구조로 되어 있으며 다수의 SQL문을 한번에 처리할수 있으며 재사용이 가능함
// 모듈화가 가능하며 변수, 예외 등의 설정이 가능함
// 변수는 기본적으로 정적 선언이지만 동적 선언이 가능함
// 저장 프로시저 : PL/SQL 언어로 쓰여지고 데이터베이스 범위 내에서 컴파일 및 실행되는 명령어 세트
// 명령어 세트는 컴파일 과정을 거쳐서 저장할 수 있으며 재사용시에는 구문검사를 하지 않으므로 실행속도가 빠름

// 저장 프로시저 구성
// 파라미터 이름  IN/OUT  데이터타입
// IN : 프로시저 내부에서 사용되는 매개변수 (프로그래밍의 변수 개념과 비슷)
// OUT : 프로시저 외부에서 전달해서 사용하지만 프로시저 호출후에도 외부에서 유지 (프로그래밍의 리턴값 개념과 비슷)
// 변수에 값 대입하는 방법 : 변수이름 := 값;
// 데이터타입은 오라클의 일반 데이터타입을 사용 가능

// PL/SQL은 .sql로 별도 저장 권장
// 저장 프로시저 자체는 에러를 내장하지 않으므로 별도 에러 메세지로 에러의 발생을 알리고 리턴에 에러코드를 저장

// OUT 파라미터는 변수 선언만 하고 함수에 대입하여 함수가 종료되더라도 그 값이 유지되어 출력이 가능해짐
// 함수의 결과값만을 담기 위한 변수 개념

// Exception으로 예외 상황을 선언 가능
// Raise를 사용하면 예외를 의도적으로 발생시킬 수 있음


create or replace procedure sp_department_select
(
/** 파라미터 선언 */
-- 참조 파라미터 선언
o_result    out number,
o_recordset out sys_refcursor
)
/** SP 내부에서 사용할 변수 선언 */
IS
-- 여기서는 사용하지 않음

/** 구현할 SQL 구문 작성 */
begin
-- 학과 목록 조회하기 -> 조회 결과를 o_recordset에 저장한다.
open o_recordset for
    select deptno, dname, loc from department order by deptno asc;
-- 결과값을 성공(=0)으로 설정
o_result := 0;

/** 예외처리 */
exception
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
end sp_department_select;
/
// 특정 값(학과)에 대한 조회를 위한 저장 프로시저 정의

-- 변수선언
var v_result number;
var v_rs refcursor;

-- 프로시저 호출
-- 파라미터로 전달된 참조변수는 프로시저 안에서의 변경사항이 프로시저 밖에서 유지된다.
execute sp_department_select(:v_result, :v_rs);

--출력하기
print v_result;
print v_rs;
// sp_department_select 실행 및 결과 확인


create or replace procedure sp_department_select_item
(
/** 파라미터 선언 */
-- 일반 파라미터
o_deptno    in  number,
-- 참조 파라미터
o_result    out number,
o_recordset out sys_refcursor
)
/** SP 내부에서 사용할 변수 선언 */
IS
-- 예외 선언
t_input_exception   exception;

/** 구현할 SQL 구문 작성 */
begin
-- 파라미터를 검사해서 필수값이 Null이라면 강제로 예외를 발생시킴.
-- > 프로시저의 제어가 Exception 블록으로 넘어감.
if o_deptno is null then
    raise t_input_exception;
end if;

-- 학과 목록 조회하기 -> 조회 결과를 o_recordset에 저장한다.
open o_recordset for
    select deptno, dname, loc from department
    where deptno = o_deptno
    order by deptno asc;
    
-- 결과값을 성공(=0)으로 설정
o_result := 0;
/** 예외처리 */
exception
    when t_input_exception then
        o_result := 1;
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
end sp_department_select_item;
/
// 특정 학과의 데이터를 조회하는 프로시저 정의

var v_result number;
var v_rs refcursor;

execute sp_department_select_item(101, :v_result, :v_rs);

print v_result;
print v_rs;
// sp_department_select_item 호출 및 결과 확인

create or replace procedure sp_department_insert
(
/** 파라미터 선언 */
-- 일반 파라미터
o_dname in  department.dname%type,
o_loc   in  department.loc%type,
-- 참조 파라미터
o_result    out number,
o_deptno    out department.deptno%type
)

/** SP 내부에서 사용할 변수 선언 */
IS

-- 예외를 선언
t_input_exception exception;
/** 구현할 SQL문 작성 */
begin
-- 파라미터 검사
if o_dname is null then
    o_deptno := 0; -- 예외 처리 전, 일련번호 값을 0으로 강제 설정
    raise t_input_exception;
end if;

-- 저장된 일련번호 채집하기 -> 조회결과를 o_deptno에 저장
select seq_department.nextval into o_deptno from dual;

-- 학과 정보 수정하기
insert into department (deptno, dname, loc)
values (o_deptno, o_dname, o_loc);

-- 결과값을 성공(=0)으로 설정
o_result := 0;

-- 모든 처리가 종료되었으므로, 변경사항을 커밋한다.
commit;

/** 예외처리 -> 예외가 발생하였으므로, 변경사항을 롤백한다. */
exception
    when t_input_exception then
        o_result := 1;
        rollback;
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
        rollback;
        
end sp_department_insert;
/
// 특정 값(학과)의 저장을 위한 저장 프로시저 정의
// 값을 생성하였으면 그 결과를 확인하기 위해서 Primary Key를 반환받을 변수를 선언함
// 데이터의 수정에 대해서는 에러가 발생할 경우 해당 작업을 취소해야 하므로 rollback을 예외 상황에 추가

var v_result number;
var v_seq number;

execute sp_department_insert('정보통신학과', '6호관', :v_result, :v_seq);
print v_result;
print v_seq;
// sp_department_insert 실행 및 결과 확인
select * from department;


create or replace procedure sp_department_update
(
/** 파라미터 선언 */
-- 일반 파라미터
o_deptno    in  department.deptno%type,
o_dname     in  department.dname%type,
o_loc       in  department.loc%type,
-- 참조 파라미터
o_result    out number,
o_rowcount  out number
)

/** SP 내부에서 사용할 변수 선언 */
is
    t_input_exception   exception; -- 파라미터가 충족되지 않은 경우
    t_data_not_found    exception; -- 입력, 수정, 삭제 된 행의 수가 0인 경우
    
/** 구현할 SQL 구문 작성 */
begin
-- 파라미터 검사
    if  o_deptno is null or o_dname is null then
        raise t_input_exception;
    end if;

-- 학과 정보 수정하기
update department set dname = o_dname, loc = o_loc where deptno = o_deptno;

-- 수정된 행의 수를 조회하기
o_rowcount := sql%rowcount;

-- 수정된 셀의 행이 없다면 강제로 에러 발생
if o_rowcount < 1 then
    raise t_data_not_found;
end if;

-- 결과값을 성공(=0)으로 설정
o_result := 0;

-- 모든 처리가 종료되었으므로, 변경사항을 커밋한다.
commit;

/** 예외처리 */
exception
    when t_input_exception then
        o_result := 1;
        rollback;
    when t_data_not_found then
        o_result := 2;
        rollback;
    when others then
        raise_application_error(-20001, sqlerrm);
        rollback;

end sp_department_update;
/
// 특정 값의 수정을 위한 저장 프로시저 정의
// 특정 개수의 데이터를 수정했는지 확인하기 위하여 수정한 데이터의 개수를 담는 변수를 선언 (SQL%ROWCOUNT에 해당 값이 기록됨)
// 함수를 실행했을때 수정된 데이터가 0개라면 입력이 잘못되었기 때문에 예외로 처리함

var v_result number;
var v_rowcount number;

execute sp_department_update(202, '컴퓨터공학과', '공학관', :v_result, :v_rowcount);

print v_result;
print v_rowcount;
// sp_department_update 실행 및 결과 확인


create or replace procedure sp_department_delete
(
/** 파라미터 선언 */
-- 일반 파라미터
o_deptno    in  department.deptno%type,
-- 참조 파라미터
o_result    out number,
o_rowcount  out number
)

/** SP 내부에서 사용할 변수 선언 */
is
    t_input_exception   exception; -- 파라미터가 충족되지 않은 경우
    t_data_not_found    exception; -- 입력, 수정, 삭제 된 행의 수가 0인 경우
    
/** 구현할 SQL 구문 작성 */
begin
--파라미터 검사
    if o_deptno is null then
        raise t_input_exception;
    end if;

-- 학과 정보 수정하기
delete from department where deptno = o_deptno;

-- 수정된 행의 수를 조회하기
o_rowcount := sql%rowcount;

-- 수정된 행이 없다면 강제로 에러 발생
if o_rowcount < 1 then
    raise t_data_not_found;
end if;

-- 결과값을 성공(=0)으로 설정
o_result := 0;

-- 모든 처리가 종료되었으므로, 변경사항을 커밋한다.
commit;

/** 예외처리 */
exception
    when t_input_exception then
        o_result := 1;
        rollback;
    when t_data_not_found then
        o_result := 2;
        rollback;
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
        rollback;
        
end sp_department_delete;
/
// 특정 값의 삭제를 위한 저장 프로시저 정의

var v_result number;
var v_rowcount number;

execute sp_department_delete(307, :v_result, :v_rowcount);

print v_result;
print v_rowcount;
// sp_department_delete 실행 및 결과 확인

// 저장된 프로시저는 프로시저 탭에 저장됨

// 인덱스 : 데이터의 검색을 빠르게 하기 위하여 행별로 고유번호를 붙임, 데이터의 수정이 발생할때마다 인덱스를 다시 재구성하기 때문에 수정속도가 느려짐
// 인덱스보다는 Primary Key의 사용이 많음
// 데이터 딕셔너리 : 데이터의 기본정보를 담는 일종의 메타데이터