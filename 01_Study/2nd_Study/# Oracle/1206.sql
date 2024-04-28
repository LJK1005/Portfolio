// INSERT �� : ���̺� �����͸� �߰��ϴ� ����
// ���ڴ� Ȭ����ǥ ���, �� �ܴ� Ȭ����ǥ ���
// �����͸� �����ϴ� ������ �÷��� �°� (�÷� ��Ī�� ������Ÿ���� desc ���̺��̸�; ���� Ȯ�� ����)


insert into student values(
    10110, 'ȫ�浿', 'khd85', 1, '851011143098',
    TO_DATE('2013-01-01 11-42-30', 'YYYY-MM-DD HH24:MI:SS'),
    '055)-777-7777', 170, 70, 101, 9903);
// �÷� ���� ������� �ʾ� �÷��� ������ ���Ѿ� ��    

insert into student(
    studno, name, userid, grade, idnum, birthdate,
    tel, height, weight, deptno, profno
) values (
    10111, '�Ѹ�', 'dooly', 2, '8202021334765',
    to_date('2011-10-01 11:42:30', 'YYYY-MM-DD HH24:MI:SS'),
    '055)-777-7777', 170, 70, 101, 9903
);
// �÷� ���� ����ϸ� ������ ������� ������ �Է� ����
select studno, name, grade from student;

desc department;
// DESC�� Null?�� NOT NULL�̶�� ��õǸ� ����ġ�� ������� ����, ���ٸ� ����ġ ���

insert into department (deptno, dname) values(210, '�������');
insert into department (deptno, dname, loc) values (211, '�����а�', NULL);
// �÷����� ����Ҷ� �÷��� ���ų� �÷��� ����ϰ� ���� Null�� �Է��ϸ� �ΰ� �Է� ����

insert into professor(
    profno, name, userid, position, sal, hiredate, deptno
) values (
    9920, '��浿', 'gilldong', '����', 450,
    to_date('2014-01-01', 'YYYY-MM-DD'), 102
);
// ��¥ ������ �Է½ÿ��� TO_DATE() �Լ��� ����Ͽ� 'YYYY-MM-DD HH24:MI:SS' Ȥ�� 'YYYY-MM-DD'�� ������ ��ȯ�Ͽ� �Է�

insert into professor(
    profno, name, userid, position, sal, hiredate, comm, deptno
) values (
    9921, '�ֶ���', 'ttun', '����', 550, sysdate, null, 102
);
// ����ð��� SYSDATE ���
select * from professor;

insert into department values (211, '�����а�2', NULL);
// �⺻Ű(Primary Key) : �÷����� �ߺ��� ������� �ʴ� �ɼ�(������ ���Ἲ), �Ϲ������� �Ϸù�ȣ(�й�, ������ȣ ��) �ʵ忡 ����
// �⺻Ű �ɼ��� �ɾ �ߺ��� ������� �ʴ� ���� ���Ἲ ���� �����̶�� �θ� (�⺻Ű ���� ����)
// �й� ���� �⺻Ű�� ���ڰ� ���������� �þ�� ��찡 ��κ��̶� Sequential�̶�� Ī�ϱ⵵ �� (Seq) Ȥ�� ID


insert into department(deptno, dname, loc)
values(seq_department.nextval, '�����а�', '5ȣ��');
// �ڵ� ���� �Ϸù�ȣ �Ӽ� : create sequence�� �̿��Ͽ� insert ������ �ڵ����� ���� �����ϵ��� �����͸� �Է��� �� �־� �������� �������� ������ �� ����
// create sequence �������̸� start with ���۰� increment by ������;
// ������ ȣ���� insert������ �������̸�.nextval�� �����

update professor set position='������', sal = 250
where profno = 9903;
// update �� : ������ ����, ���� ������ �÷��� �����Ͽ� ���� ����
select name, position, sal from professor where profno = 9903;

delete from student where studno = 20103;
// delete �� : ������ ����
select studno, name from student where studno = 20103;


update student set name = '�ֶ���';
delete from student;
// update�� delete ���� where ���� ������� ������ ��ü �������� �����̳� ������ �߻��ϹǷ� ���� �ʿ�


update student set birthdate = sysdate where studno = 20101;
// ������ �Է�, ����, �����ÿ��� �Լ��� Ȱ���� �� ����
// ��¥�� �����ϰ�� �Լ��� ����ϴ� ���� ���� ����
select name, to_char(birthdate, 'YYYY-MM-DD')
from student where studno = 20103;

commit;
rollback;
// ����Ŭ���� INSERT, UPDATE, DELETE�� ����� ���������� ������ �ݿ��� ���� �ʱ� ������ ���� ������ ���� COMMIT �ϰų� ROLLBACK���� ��Ұ� ������ (Ʈ������)
// MySQL������ �⺻ ������ AUTO-COMMIT (Ʈ������X)�̱� ������ �ѹ� �߸��� ������ �Ǹ� �ǵ����� ��ƴ�
// COMMIT�� �ϸ� �ǵ��� �� ����


// ��������1
desc department;
insert into department (deptno, dname, loc) values (seq_department.nextval, '���ͳ�������', '���а�');
select * from department;

// ��������2
update department set loc = '5ȣ��' where loc = '���а�';
select * from department;

// ��������3
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
// CREATE�� : ���̺� ����, DDL
// �÷� �̸��� ������ Ÿ��, Null ��뿩��(�Ⱦ��� ���), Primary Key �÷��� ����
// Primary Key�� ���� Primary Key(�÷��̸�)�� ���ų� Ư�� �÷� �ڿ� Primary Key�� �߰�
// Primary Key ������ ���Ͽ� �������� ����
// Primary Key�� Null�� ������� ����


// �ؽ�Ʈ Ÿ��
// varchar2 : ���� ����, ������ ���� ������ �Է� ���� ���̿� ���߾� �������� ũ�Ⱑ ����
// char : �Է��� ���ڿ� ���þ��� ������ ���̷� ������, ���� ����� �ʰ��� ��� ������ ���ų� �ؽ�Ʈ�� �ڸ� (�����ȣ, �ֹι�ȣ �� ������ ���̿� ���Ͽ� ���)
// long : ���� ����, �� ���ڿ� (�Խ��� ���� ��)�� �ַ� ���

// ������
// number : ���� ��ȣ�� �������� ������ ����, (��ü�ڸ�, �Ҽ����ڸ�)�� �ٿ��� �Ǽ��� �����͵� �Է� ����

// date : ��¥��

drop table mytable;
// ���̺� ������ drop���� �����ϸ�, Ʈ�������� �ƴϱ� ������ �ѹ� �����ϸ� ������ �� ����.
// alter table : ���̺� ����, ������ �������� �ڿ� ����

alter table mytable rename to mytest;
// ���̺� �̸� ����

alter table mytest add edit_date date not null;
// �÷� �߰�
// �̹� �����Ͱ� �ִ� ���̺� �÷��� �߰��� ���� Not Null�δ� ������ �� ���� (���� �����Ϳ� ���ؼ��� Null�� �Ǿ� �ϹǷ�)
// Not Null�� �߰��ؾ� �Ѵٸ� �켱 Null�� �߰��Ͽ� ��� �����͸� ������Ʈ �ϰ� �÷��� Not Null�� �����ؾ� ��
desc mytest;

alter table mytest rename column memo to content;
// �÷� �̸� ����
desc mytest;

alter table mytest modify(content varchar2(512) not null);
// �÷��� ������Ÿ�� ����
// ������ �ս��� �����ϱ� ���� �÷��� ��������� ���� �� ���� (Null�� �ְų� ���� ���ٸ� ����)
// Null�� �ִ� �÷��� ������ ������ ������ �� ����
// Null�� ���� ��쿡�� Not Null�� ���� ����

alter table mytest drop column edit_date;
// �÷� ����

alter table mytest drop primary key;
// �⺻Ű �Ӽ� ����

alter table mytest add constraint pk_mytest_id primary key(id);
// �⺻Ű �Ӽ� ����
// �÷� �������� �⺻Ű �Ӽ��� �ο��� ��� �Ӽ� �̸��� �����ؾ��� (Create������ �ڵ� ������)


// ���̺��� ��������
// � �����Ͱ� �ٸ� ���̺��� �÷��� �ִ� �����Ϳ��� �� ��� (���� ���̺��� �а���ȣ�� �а� ���̺� �־�� ��) �ش� ���̺��� �����Ͽ��� �� (���� ���Ἲ ���� ����)
// Ư�� �÷��� �ٸ� ���̺��� �⺻Ű�� �����ϴ� ��찡 ���� ���� (����Ű, �ܷ�Ű, Foreign Key�� Ī��)
// Primary Key ���� ������ �Ǵ� ��� 1:1 �����̸�, Primary Key - �Ϲ� Į���� 1:n�� ���� (ī��θ�Ƽ)
// ������ ����ȭ : �������� ȿ������ ���Ͽ� ����Ű�� �̿��� �����͸� �ɰ����� ���� (ex. �а���(+a)�� �� �л����� ������ �ο����� �ʰ� �а� ��ȣ�� �ο��Ͽ� ȿ������ �ø�)
// Primary Key ���� 1:1 ������ ��ǻ� �����͸� ��ĥ�� ���� (������ȭ)

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
// ����Ű ����
// ������ �޴� �����ʹ� �����ϴ� �����Ͱ� ��� �����Ǳ� �������� ������ �Ұ��� (ex. �а� �� �л��� �ΰ� �а��� �����ϴ°��� �Ұ�)

insert into grade (id, studno, subject, point, reg_date)
values (seq_grade.nextval, 1234, '�����ͺ��̽�', 98, sysdate);
// ���� ��� �����Ͱ� ���� ��� �Է� �õ��� ���� �߻�

insert into grade(id, studno, subject, point, reg_date)
values (seq_grade.nextval, 10101, '�����ͺ��̽�', 98, sysdate);
// ���� ��� �ִ� �����͸� ����Ͽ� ������ �߰�

delete from student where studno = 10101;
// ���� ��� �ִ� ���� �����ʹ� �����ϴ� �����͸� �����ϱ� �������� ������ �� ����
// �⺻Ű / ����Ű�� ���� ������ �ڰ��� ���迡 ���� ������
delete from grade where studno = 10101;
delete from student where studno = 10101;
commit;

select * from grade;

// PL/SQL : ����Ŭ�� ��ü������ ����� ���
// ��� ������ �Ǿ� ������ �ټ��� SQL���� �ѹ��� ó���Ҽ� ������ ������ ������
// ���ȭ�� �����ϸ� ����, ���� ���� ������ ������
// ������ �⺻������ ���� ���������� ���� ������ ������
// ���� ���ν��� : PL/SQL ���� �������� �����ͺ��̽� ���� ������ ������ �� ����Ǵ� ��ɾ� ��Ʈ
// ��ɾ� ��Ʈ�� ������ ������ ���ļ� ������ �� ������ ����ÿ��� �����˻縦 ���� �����Ƿ� ����ӵ��� ����

// ���� ���ν��� ����
// �Ķ���� �̸�  IN/OUT  ������Ÿ��
// IN : ���ν��� ���ο��� ���Ǵ� �Ű����� (���α׷����� ���� ����� ���)
// OUT : ���ν��� �ܺο��� �����ؼ� ��������� ���ν��� ȣ���Ŀ��� �ܺο��� ���� (���α׷����� ���ϰ� ����� ���)
// ������ �� �����ϴ� ��� : �����̸� := ��;
// ������Ÿ���� ����Ŭ�� �Ϲ� ������Ÿ���� ��� ����

// PL/SQL�� .sql�� ���� ���� ����
// ���� ���ν��� ��ü�� ������ �������� �����Ƿ� ���� ���� �޼����� ������ �߻��� �˸��� ���Ͽ� �����ڵ带 ����

// OUT �Ķ���ʹ� ���� ���� �ϰ� �Լ��� �����Ͽ� �Լ��� ����Ǵ��� �� ���� �����Ǿ� ����� ��������
// �Լ��� ��������� ��� ���� ���� ����

// Exception���� ���� ��Ȳ�� ���� ����
// Raise�� ����ϸ� ���ܸ� �ǵ������� �߻���ų �� ����


create or replace procedure sp_department_select
(
/** �Ķ���� ���� */
-- ���� �Ķ���� ����
o_result    out number,
o_recordset out sys_refcursor
)
/** SP ���ο��� ����� ���� ���� */
IS
-- ���⼭�� ������� ����

/** ������ SQL ���� �ۼ� */
begin
-- �а� ��� ��ȸ�ϱ� -> ��ȸ ����� o_recordset�� �����Ѵ�.
open o_recordset for
    select deptno, dname, loc from department order by deptno asc;
-- ������� ����(=0)���� ����
o_result := 0;

/** ����ó�� */
exception
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
end sp_department_select;
/
// Ư�� ��(�а�)�� ���� ��ȸ�� ���� ���� ���ν��� ����

-- ��������
var v_result number;
var v_rs refcursor;

-- ���ν��� ȣ��
-- �Ķ���ͷ� ���޵� ���������� ���ν��� �ȿ����� ��������� ���ν��� �ۿ��� �����ȴ�.
execute sp_department_select(:v_result, :v_rs);

--����ϱ�
print v_result;
print v_rs;
// sp_department_select ���� �� ��� Ȯ��


create or replace procedure sp_department_select_item
(
/** �Ķ���� ���� */
-- �Ϲ� �Ķ����
o_deptno    in  number,
-- ���� �Ķ����
o_result    out number,
o_recordset out sys_refcursor
)
/** SP ���ο��� ����� ���� ���� */
IS
-- ���� ����
t_input_exception   exception;

/** ������ SQL ���� �ۼ� */
begin
-- �Ķ���͸� �˻��ؼ� �ʼ����� Null�̶�� ������ ���ܸ� �߻���Ŵ.
-- > ���ν����� ��� Exception ������� �Ѿ.
if o_deptno is null then
    raise t_input_exception;
end if;

-- �а� ��� ��ȸ�ϱ� -> ��ȸ ����� o_recordset�� �����Ѵ�.
open o_recordset for
    select deptno, dname, loc from department
    where deptno = o_deptno
    order by deptno asc;
    
-- ������� ����(=0)���� ����
o_result := 0;
/** ����ó�� */
exception
    when t_input_exception then
        o_result := 1;
    when others then
        raise_application_error(-20001, sqlerrm);
        o_result := 9;
end sp_department_select_item;
/
// Ư�� �а��� �����͸� ��ȸ�ϴ� ���ν��� ����

var v_result number;
var v_rs refcursor;

execute sp_department_select_item(101, :v_result, :v_rs);

print v_result;
print v_rs;
// sp_department_select_item ȣ�� �� ��� Ȯ��

create or replace procedure sp_department_insert
(
/** �Ķ���� ���� */
-- �Ϲ� �Ķ����
o_dname in  department.dname%type,
o_loc   in  department.loc%type,
-- ���� �Ķ����
o_result    out number,
o_deptno    out department.deptno%type
)

/** SP ���ο��� ����� ���� ���� */
IS

-- ���ܸ� ����
t_input_exception exception;
/** ������ SQL�� �ۼ� */
begin
-- �Ķ���� �˻�
if o_dname is null then
    o_deptno := 0; -- ���� ó�� ��, �Ϸù�ȣ ���� 0���� ���� ����
    raise t_input_exception;
end if;

-- ����� �Ϸù�ȣ ä���ϱ� -> ��ȸ����� o_deptno�� ����
select seq_department.nextval into o_deptno from dual;

-- �а� ���� �����ϱ�
insert into department (deptno, dname, loc)
values (o_deptno, o_dname, o_loc);

-- ������� ����(=0)���� ����
o_result := 0;

-- ��� ó���� ����Ǿ����Ƿ�, ��������� Ŀ���Ѵ�.
commit;

/** ����ó�� -> ���ܰ� �߻��Ͽ����Ƿ�, ��������� �ѹ��Ѵ�. */
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
// Ư�� ��(�а�)�� ������ ���� ���� ���ν��� ����
// ���� �����Ͽ����� �� ����� Ȯ���ϱ� ���ؼ� Primary Key�� ��ȯ���� ������ ������
// �������� ������ ���ؼ��� ������ �߻��� ��� �ش� �۾��� ����ؾ� �ϹǷ� rollback�� ���� ��Ȳ�� �߰�

var v_result number;
var v_seq number;

execute sp_department_insert('��������а�', '6ȣ��', :v_result, :v_seq);
print v_result;
print v_seq;
// sp_department_insert ���� �� ��� Ȯ��
select * from department;


create or replace procedure sp_department_update
(
/** �Ķ���� ���� */
-- �Ϲ� �Ķ����
o_deptno    in  department.deptno%type,
o_dname     in  department.dname%type,
o_loc       in  department.loc%type,
-- ���� �Ķ����
o_result    out number,
o_rowcount  out number
)

/** SP ���ο��� ����� ���� ���� */
is
    t_input_exception   exception; -- �Ķ���Ͱ� �������� ���� ���
    t_data_not_found    exception; -- �Է�, ����, ���� �� ���� ���� 0�� ���
    
/** ������ SQL ���� �ۼ� */
begin
-- �Ķ���� �˻�
    if  o_deptno is null or o_dname is null then
        raise t_input_exception;
    end if;

-- �а� ���� �����ϱ�
update department set dname = o_dname, loc = o_loc where deptno = o_deptno;

-- ������ ���� ���� ��ȸ�ϱ�
o_rowcount := sql%rowcount;

-- ������ ���� ���� ���ٸ� ������ ���� �߻�
if o_rowcount < 1 then
    raise t_data_not_found;
end if;

-- ������� ����(=0)���� ����
o_result := 0;

-- ��� ó���� ����Ǿ����Ƿ�, ��������� Ŀ���Ѵ�.
commit;

/** ����ó�� */
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
// Ư�� ���� ������ ���� ���� ���ν��� ����
// Ư�� ������ �����͸� �����ߴ��� Ȯ���ϱ� ���Ͽ� ������ �������� ������ ��� ������ ���� (SQL%ROWCOUNT�� �ش� ���� ��ϵ�)
// �Լ��� ���������� ������ �����Ͱ� 0����� �Է��� �߸��Ǿ��� ������ ���ܷ� ó����

var v_result number;
var v_rowcount number;

execute sp_department_update(202, '��ǻ�Ͱ��а�', '���а�', :v_result, :v_rowcount);

print v_result;
print v_rowcount;
// sp_department_update ���� �� ��� Ȯ��


create or replace procedure sp_department_delete
(
/** �Ķ���� ���� */
-- �Ϲ� �Ķ����
o_deptno    in  department.deptno%type,
-- ���� �Ķ����
o_result    out number,
o_rowcount  out number
)

/** SP ���ο��� ����� ���� ���� */
is
    t_input_exception   exception; -- �Ķ���Ͱ� �������� ���� ���
    t_data_not_found    exception; -- �Է�, ����, ���� �� ���� ���� 0�� ���
    
/** ������ SQL ���� �ۼ� */
begin
--�Ķ���� �˻�
    if o_deptno is null then
        raise t_input_exception;
    end if;

-- �а� ���� �����ϱ�
delete from department where deptno = o_deptno;

-- ������ ���� ���� ��ȸ�ϱ�
o_rowcount := sql%rowcount;

-- ������ ���� ���ٸ� ������ ���� �߻�
if o_rowcount < 1 then
    raise t_data_not_found;
end if;

-- ������� ����(=0)���� ����
o_result := 0;

-- ��� ó���� ����Ǿ����Ƿ�, ��������� Ŀ���Ѵ�.
commit;

/** ����ó�� */
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
// Ư�� ���� ������ ���� ���� ���ν��� ����

var v_result number;
var v_rowcount number;

execute sp_department_delete(307, :v_result, :v_rowcount);

print v_result;
print v_rowcount;
// sp_department_delete ���� �� ��� Ȯ��

// ����� ���ν����� ���ν��� �ǿ� �����

// �ε��� : �������� �˻��� ������ �ϱ� ���Ͽ� �ະ�� ������ȣ�� ����, �������� ������ �߻��Ҷ����� �ε����� �ٽ� �籸���ϱ� ������ �����ӵ��� ������
// �ε������ٴ� Primary Key�� ����� ����
// ������ ��ųʸ� : �������� �⺻������ ��� ������ ��Ÿ������