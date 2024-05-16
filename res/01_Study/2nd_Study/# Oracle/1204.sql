SELECT * FROM department;

SELECT deptno, dname from department;

select deptno from student;
select distinct deptno from student; // distinct�� ���̸� �ߺ��� ���ŵ�

select deptno, grade from student;
select distinct deptno, grade from student; // �ټ� �÷��� ���Ͽ� distinct ����� ��� ���� ��ġ�ؾ� �ߺ����� ó����

select dname �а��̸�, deptno �а���ȣ from department; // �÷��� ��Ī �ο�, Ȭ����ǥ�� ���� ����
select dname as �а��̸�, deptno as �а���ȣ from department; // as�� ���� ����, �������� ������

select name, sal, sal*12+100 from professor; // select ������ ��������� �����ϸ�, ��ȸ ����� ����Ǿ� ��µ�, select ������ ���� �����ʹ� ����� ������ ����
// sql���� ��ȸ�ÿ��� ������ ���� ������ ���� �÷��� ���̽����� ������ ��� ��Ī ������ �ʼ�����

// select���� where�� �߰��Ͽ� ��ȸ ������ ������ �� ����
// �񱳿�����(=, > ��), ��������(AND, OR ��) ��� ����

select studno, name, deptno from student where grade=1; // grade(�г�)�� 1�� ������ ��ȸ

select studno, name, grade, deptno, weight from student
where weight <= 70; // SQL�������� �� ��ɾ �����ݷ�(;)���� �����ϹǷ� ���̽�� �޸� �ٹٲ�, �鿩���� ���� ����� �����ο�

select name, studno, grade, weight, deptno from student where grade=1 and weight >= 70; // AND ���

// ��������1
select name, studno from student;

// ��������2
select distinct position from professor;

// ��������3
select deptno as �μ�, dname as �μ���, loc as ��ġ from department;

// ��������4
select name as �л��̸�, (height-110)*0.9 as ǥ��ü�� from student;

// ��������5
select studno, name, grade from student where deptno=101;

// ��������6
select profno, name, sal from professor where deptno=101;

// ��������7
select studno, name, grade, deptno, height from student where height >= 170;

// ��������8
select name, studno, grade, weight, deptno from student where grade=1 or weight >= 70;




