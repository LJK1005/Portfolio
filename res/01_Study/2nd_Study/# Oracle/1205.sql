select studno, name, weight from student
where weight between 50 and 70; // BETWEEN �����ڴ� Ư�� ������ �����͸� ��ȸ ���� (�̻�/���� ����)
select studno, name, weight from student
where weight >= 50 and weight <= 70; // ���� ���� ȿ��

select name, grade, deptno from student
where deptno in (102, 201); // IN �����ڴ� ��ȣ ���� ���� ������ �ȿ� �ִ� �Ͱ� ��ġ�ϴ� ���� ������ ���� ��
select name, grade, deptno from student
where deptno = 102 or deptno = 201; // ���� ���� ȿ��

select name, grade, deptno from student where name like '��%'; // LIKE �����ڴ� ���ڿ��� Ư���Ͽ� ���� ����
// %�� �ش� �ڸ����� ��� ���ڿ��̶� ������ �ǹ�

// NULL : ��Ȯ�� ������, ���� �������� ���� ���� �ǹ� (����ġ)
// ���� 0�̳� ������� �ٸ� ����

select name, position, comm from professor; // COMM�� null �� ����

select name, position, comm from professor where comm is null; // COMM�� null�� ������ ��ȸ
select name, position, comm from professor where comm is not null; // COMM�� null�� �ƴ� ������ ��ȸ

select name, grade, deptno from student
where depno = 102 and (grade = 4 or grade = 1); // AND�� OR���� �켱�ǳ� ��ȣ�� ��� OR�� ���� �����

// ��������1
select * from professor where sal between 300 and 400;

// ��������2
select profno, name, position, deptno from professor where position in ('������', '���Ӱ���');

// ��������3
select deptno, dname, loc from department where dname like '%����%';
// like�� �˻� ������ ���Ƽ� ES(Elastic Search)��� �ڿ��� �˻������� ���⵵ ��

// ��������4
select studno, name, grade, profno from student where profno is not null;

// ��������5
select name, grade, deptno from student where (deptno = 102 and grade = 4) or grade = 1; // ��ȣ�� ��� �� (AND�� OR���� �켱��)



select name, grade, tel from student order by name;
// ORDER BY ���� WHERE �� �ڿ� ��, ��������(ASC)�� ����Ʈ, DESC�� ���� ��� �ʿ�

select studno, name, grade, deptno, userid from student
order by deptno asc, grade desc;
// ,�� �̿��Ͽ� ORDER BY ���� �������� ���� �ɼ��� ������ �� ����. ���� ����� �������Ǻ��� ���������� ����

select rownum as RNUM, tbl.* from(
    select deptno, dname from department order by deptno asc) tbl;
// ROWNUM : �����͸� �ѹ��� ��ȸ�ϸ� �ӵ��� �������Ƿ� ������ �κ� ��ȸ�� ���Ͽ� �������(select���� ���) �� ���ڵ忡 ���� ������ �ο���
// �ش� ������ �������� ���� ����Ʈ ��ǰ ��ȸ ȭ�� ��� �������� �����Ͽ� �����Ͱ� ��µ�
// SELECT�� 2������ ���, 1�� SELECT ���� ��Ī�� ���̰� �ش� ��Ī�� 2�� SELECT ������ ���
// MySQL�� LIMIT ���� ���� ������ ���
// �κ���ȸ�� ROWNUM�� ���� WHERE ���� �̿��Ͽ� ������ �ɾ ���
// ROWNUM�� 0�� �ƴ� 1���� �����ϱ� ������ ��ȸ ���� ��뿡 ���ǰ� �ʿ�
// ���̽� �����̽��� n�̻�, m�̸��� ����̸� ROWNUM�� n�ʰ�, m������ ������� �����ؾ� ������ ����� ��
// ROWNUM�� Ư�� ���̰ų� Ư�� �� �̻�/�ʰ��� ������ �� ��� �� ������ ��ȸ�� ���� ������ 1�� SELECT�� ROWNUM�� Ȯ���ϰ� �ش� ROWNUM�� 2�� SELECT�� ������ �ɾ� ��ȸ
// ex) ROWNUM = 5�� �׳� ���� ��ȸ ����� 1���ε� ROWNUM 5�� ã�� ���̹Ƿ� ���� ��ȸ�� �ȵ�, ����/�̸��� ���� ��ȸ ����

select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc
    )tbl where rownum <= 3
) where RNUM > 0;

// ��������1
select name, grade, idnum from student order by grade desc;

// ��������2
select name, grade, deptno from student where deptno = 101 order by birthdate;

// ��������3
select name, studno, grade from student order by grade asc, name asc;

// ��������4
select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc) tbl
) where RNUM = 5;

select * from(
    select rownum as RNUM, tbl.* from(
        select name, position, sal from professor order by sal desc
    )tbl where rownum <= 5
) where RNUM > 4;


// �Լ��� SQLD���� ����Ŭ ���� �Լ��� ���� ����
// SQL���� �Լ� input�� �÷�����


select name, length(name) from student; // length �Լ��� �ش� ������ ���� ���� ��ȸ

select name, substr(name, 1, 1) from student;
// substr(�÷�, �����ε���, ��ȸ�� ���ڰ���), ����Ŭ�� ���̽�� �޸� 1���� count �ϴ°Ϳ� ����

select name, substr(name, length(name)) from student;
// length(name)�� �����ϸ� ������ ���� ���� ����, stbstr���� ������ �Ķ���� �������� ������ ������ �����̽���

select name, substr(name, 2, 1) from student;

select name, replace(name, '��', 'lee') from student;
// replace(���ڿ�, �������ڿ�, �����ҹ��ڿ�) : ������ �״�� ���


select concat(name, grade) from student;
select name || grade from student;

select concat(concat(concat(name, ' '), grade), '�г�') from student;
select name || ' ' || grade || '�г�' from student;
// ����Ŭ�� concat �Լ��� input�� �Ķ���Ͱ� 2���� �����ϹǷ� concat�� ������ ����ؾ� ��
// concat() ��� ||�� ����Ͽ� ���ڿ��� ������

select trim(name), ltrim(name), rtrim(name) from student;
// trim, ltrim, rtrim : ��������, ���̽��� strip�� ����

select instr(name, '��'), name from student;
// instr(���ڿ�, ã����) : ���̽��� find. Ư�� ���ڿ��� ��Ÿ���� ��ġ ��ȸ ������ 0

select upper(userid), lower(userid) from student;
// upper, lower : ��/�ҹ��� ��ȯ



select sysdate from dual;
// �����Ͱ� �ʿ���� sysdate�� ����� ��� ���� ���̺� dual�� ��ȸ�Ͽ� ������ �ϼ�

select to_char(sysdate, 'YYYYMMDDHH24MIDD') from dual;
// TO_CHAR�� ����ϸ� ���ϴ� �������� ��¥ ��ȸ ����


select to_char(sysdate+100, 'YYYY-MM-DD'), to_char(sysdate-7, 'YYYY-MM-DD') from dual;
// ��¥�� ���� ������ �� �������� ����


// ��������1
select name, replace(name, substr(name, 2, 1), '*') from student;
select name, substr(name, 1, 1) || '*' || substr(name, 3, 1) from student;

// ��������2
select name, replace(idnum, substr(idnum, 7), '*******') from student;
select name, substr(idnum, 1, 6) || '*******' from student;

// ��������3
select name, birthdate from student where birthdate >= '1980-01-01';
select name, to_char(birthdate, 'YYYY-MM-DD') from student
where to_char(birthdate, 'YYYY') > 1980;


// �׷��Լ� : ���̺��� ��ü ��, �ϳ� �̻��� �÷��� �������� �÷� ���� ���� �׷�ȭ�Ͽ� ����� ����ϴ� �Լ� (����, �ִ밪, ��� ��)

select count(studno) from student where grade = 3;
// grade�� 3�� �л� �� ��ȸ


select comm from professor where deptno = 101;
select count(comm) from professor where deptno = 101;
// count�� ���� ���� ���� �Լ��� Null ���� �����ϰ� ��

select count(*) from professor where deptno = 101;
// count �Լ��� *�� �����ϰų� null�� ���� �÷��� �����ϸ� null���� ���� ����

select max(sal) from professor;
select min(sal) from professor;
select sum(sal) from professor;
select avg(height) from student;
select avg(weight), sum(weight) from student where deptno = 101;
// �����Լ��� ���


// GROUP BY �� : Ư�� �÷� ���� �������� ���̺��� ��ü ���� �׷캰�� ����
// WHERE ���� ORDER BY �� ���̿� �� (��ġ ���� �Ұ�)
// GROUP BY ������ �ݵ�� �÷� �̸��� ���� ��
// SELECT ������ ���� �Լ� ���� ������ �÷� �̸��̳� ǥ������ GROUP BY ���� �ݵ�� ���ԵǾ� �� (�׷����� �ϰų�, ���踦 �ϰų�)

select deptno, name from professor order by deptno;
select deptno, name from professor group by deptno; // name �÷��� ó���� ������� �ʾ����Ƿ� ���� �߻�
select deptno, count(name) from professor group by deptno; // �����Լ� count ó���� �Ͽ� deptno�� ������ ������ ��ȸ

select deptno, count(*), count(comm) from professor group by deptno;


select deptno, grade, count(*), avg(weight)
from student
group by deptno, grade;
// ���� �÷��� �̿��� �׷캰 �˻�
// 2�� �̻��� �÷��� �ĸ��� ��� ���� �׷����� ����


// HAVING �� : GROUP BY ���� ���� ���ǽ�
// SELECT ���� ORDER BY ���� �����Լ��� ���� ��� SELECT ������ ��Ī�� �����ϰ� �ش� ��Ī�� ORDER BY ������ ����Ͽ� ���� �ð� ���� ����

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
// WHERE������ �����Լ��� ����� �� ����

// ��������1
select max(height), min(height) from student where deptno = 101;

// ��������2
select deptno, avg(sal), min(sal), max(sal) from professor group by deptno;

// ��������3
select deptno, avg(weight) avg_weight, count(studno) from student group by deptno order by avg_weight desc;

// ��������4
select deptno, count(profno) from professor group by deptno having count(profno) <= 2;


select name, department.deptno, dname from professor, department;
// JOIN : �� �� �̻��� ���̺��� �����Ͽ� �ʿ��� �����͸� ��ȸ�ϴ� ���
// �����ʹ� ������ �ߺ��� ���Ͽ� �����ϴ� ���� ���� (������ ���ڼ�)
// ���� �����Ϳ� �а� �����͸� ������ �����Ͽ� ������ �а��� ���ڿ��� ��ȸ�� �� �� �����͸� ���ν��� �����͸� ��ȸ
// JOIN���� 2�� �̻��� ���̺��� ��ȸ�� �� ���̺��� from ������ �ĸ��� �������� ���̺��� ��ȸ
// ���̺� ���� ������ ��Ī�� �÷��� ���� ��� �浹�� ȸ���ϱ� ���� select ������ '���̺��̸�.�÷��̸�' �������� �÷��� ���
// �ƹ� JOIN ���� ���� Ȥ�� �߸��� �������� �ټ��� ���̺��� ��ȸ�ϸ� ��� 1:1 ������ �����ϰ� �� (ī��� ��), �ش� ������ ���ؾ� ��
// ȣ���ϴ� ���̺��� ��Ī�� �ο��� �� ����

select name, department.deptno, dname from professor, department
where professor.deptno = department.deptno;

select p.name, d.deptno, d.dname from professor p, department d
where p.deptno = d.deptno;
// EQUI JOIN : ���� ��� ���̺��� ���� Į���� ���� '=' �񱳸� ����� ���� ���� ���� �ุ�� �����Ͽ� ����� �����ϴ� ����

select p.name, d.deptno, d.dname 
from professor p, department d
where p.deptno = d.deptno and p.deptno = 101;


select p.name, d.deptno, d.dname from professor p
inner join department d on p.deptno = d.deptno;

select p.name, d.deptno, d.dname from professor p
inner join department d on p.deptno = d.deptno
where p.deptno = 101;
// INNER JOIN : ����� EQUI JOIN�� ����, WHERE���� ���� �÷��� ������ �ɾ��� EQUI JOIN�� �޸� �߰��� �ҷ����� ���̺��� INNER JOIN���� ����
// WHERE�� ON���� �����Ͽ� JOIN ó��
// ��ȸ�ϰ� ���� �����͸� Ȯ���Ͽ� FROM���� �ҷ����� ���̺�� INNER JOIN���� �ҷ����� ���̺��� ����
// INNER JOIN�� �߰����� �˻� ������ �ʿ��� ��� WHERE �� ���(ON�� ����ϸ� ������ ���� ����)


select s.name, p.name from student s, professor p where s.profno = p.profno; // EQUI JOIN
select s.name, p.name from student s inner join professor p on s.profno = p.profno; // INNER JOIN
// OUTER JOIN : ���ǿ� �������� �ʴ� ��鵵 ���Խ��� ���ս�Ŵ
// LEFT OUTER : JOIN ������ ����� ���̺��� ������ ���̺� ���Ͽ� ���ǿ� �������� �ʴ� �����ͱ��� ��ȸ
// RIGHT OUTER : JOIN ������ ����� ���̺��� �������� ���̺� ���Ͽ� ���ǿ� �������� �ʴ� �����ͱ��� ��ȸ
// LEFT / RIGHT JOIN�� ������ �ٲ�� ����� ���� ������ �ϳ��� �����Ͽ� ���°� ����
// FULL OUTER : JOIN ������ ����� ��� ���̺� ���Ͽ� ���ǿ� �������� �ʴ� �����ͱ��� ��ȸ, �ý��� ���ɿ� ������ ũ�� �־� �� ���� ����


select s.name, p.name from student s left outer join professor p on s.profno = p.profno;
select s.name, p.name from student s right outer join professor p on s.profno = p.profno;
// OUTER JOIN�� �ٸ� ���̺� ����� �����Ͱ� ������ Null ���� ó���Ͽ� ������ ���̺� ���� ��� ������ ��ȸ ����

select s.name, p.name from student s left outer join professor p on s.profno = p.profno;
select s.name, p.name from student s, professor p where s.profno = p.profno(+);
// ����Ŭ������ EQUI JOIN�� WHERE ������ Ư�� ���̺�/�÷��� (+)�� �ٿ��� OUTER JOIN�� ������ �� ���� ((+)�� �����ʿ��� null���� ���)

select s.name, p.name from student s right outer join professor p on s.profno = p.profno;
select s.name, p.name from student s, professor p where s.profno(+) = p.profno; // (+) �����ڸ� �̿��� RIGHT OUTER JOIN


// �������� (SubQuery)
// �ϳ��� SQL���� ó������� �ٸ� SQL ��ɹ��� �����ϱ� ���� �� �� �̻��� SQL���� ��ȣ�� �ϳ��� SQL������ ������ ����
// ���������� �����ϴ� SQL���� ����������� Ī��

// ���� �� ��������, ���� �� ���������� ���� ��ȯ�ϴ� ���� �޶� �񱳿����ڿ� IN �������� ��� ���ΰ� �޶���

select name, position from professor
where position = (
    select position from professor where name = '������');
// ���� ����� �������� �ϴ� ��������

select name, deptno, grade, height from student
where grade = 1 and height > (
    select avg(height) from student);

select name, dname from student s
inner join department d on s.deptno = d.deptno
where s.deptno = (select deptno from student where name = '�̱���');
// JOIN�� �Բ� ����ϴ� ��������

select studno, grade, name from student
where profno in (select profno from professor where sal > 300);
// ���� �� �������� : ������� �� ������ �� �� ���� ���������� IN�� ����Ͽ� ���� ����


// ��������1
select s.studno, s.name, s.deptno, d.dname, d.loc from student s, department d where d.deptno = s.deptno;

// ��������2
select s.studno, s.name, s.deptno, d.dname, s.grade from student s, department d where d.deptno = s.deptno and s.deptno = 102;

// ��������3
select s.name, s.grade, p.name, p.position from student s, professor p where s.profno = p.profno;

// ��������4
select s.name, s.grade, p.name, p.position from student s inner join professor p on s.profno = p.profno;

// ��������5
select name, grade from student where grade = (select grade from student where userid = 'jun123');

// ��������6
select name, deptno, weight from student where weight < (
    select avg(weight) from student where deptno = 101);

// ��������7
select s.name, s.weight, d.dname, p.name from student s, department d, professor p
where s.profno = p.profno and s.deptno = d.deptno and s.weight < (
    select avg(weight) from student where deptno = (
        select deptno from student where name = '�̱���')
        ); // EQUI JOIN

select s.name, s.weight, d.dname, p.name from student s
inner join department d on s.deptno = d.deptno
inner join professor p on s.profno = p.profno
where s.weight < (
    select avg(weight) from student where deptno = (
        select deptno from student where name = '�̱���')
        ); // INNER JOIN

// ��������8
select name, grade, height from student
where grade = (select grade from student where studno = 20101)
and height > (select height from student where studno = 20101);

// ��������9
select s.studno, d.dname, s.grade, s.name from student s, department d
where s.deptno = d.deptno and d.dname like '%����%';

select studno, dname, grade, name from student s
inner join department d on s.deptno = d.deptno
where s.deptno in (select deptno from department where dname like '%����%'); // �������� ���















