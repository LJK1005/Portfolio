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