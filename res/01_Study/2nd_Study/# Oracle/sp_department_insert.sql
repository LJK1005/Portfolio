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