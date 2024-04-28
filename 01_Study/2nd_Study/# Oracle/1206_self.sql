SELECT * FROM student;

SELECT * FROM mytest;

CREATE TABLE self_table (
id	NUMBER NOT NULL,
name varchar2(50) NOT NULL,
tel varchar2(50) NOT NULL,
memo varchar2(50) NULL,
PRIMARY key(id));

CREATE SEQUENCE self_seq START WITH 100 INCREMENT BY 1;

SELECT * FROM self_table;

INSERT INTO self_table (
id, name, tel, memo) values(
self_seq.nextval, '이름1', '010-1111-1111', '메모입니다.');

INSERT INTO self_table (
id, name, tel, memo) values(
self_seq.nextval, '이름2', '010-2222-2222', '배고프다');

INSERT INTO self_table (
id, name, tel, memo) values(
self_seq.nextval, '이름3', '010-3333-3333', Null);

INSERT INTO self_table (
id, name, tel, memo) values(
self_seq.nextval, '이름4', '010-1234-1234', '배고프다');

ALTER TABLE self_table ADD att NUMBER NULL;

UPDATE self_table SET att = rownum;

ALTER TABLE self_table DROP column att;

SELECT * FROM grade;

INSERT INTO grade (id, studno, subject, point, reg_date)
values(seq_grade.nextval, 10103, '국어', 79, sysdate);

INSERT INTO GRADE values(
seq_grade.nextval, 10104, '영어', 99, sysdate);




