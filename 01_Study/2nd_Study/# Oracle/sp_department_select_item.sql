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