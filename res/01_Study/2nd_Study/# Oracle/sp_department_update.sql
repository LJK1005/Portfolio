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
