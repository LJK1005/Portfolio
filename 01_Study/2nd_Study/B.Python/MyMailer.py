# -> 경로 정보를 취득하기 위한 모듈
import os.path
# -> 발송서버와 연동하기 위한 모듈
from smtplib import SMTP
# -> 본문 구성 기능
from email.mime.text import MIMEText
# -> 파일을 Multipart 형식으로 변환
from email.mime.application import MIMEApplication
# -> 파일을 본문에 추가하는 기능 제공
from email.mime.multipart import MIMEMultipart

def sendMail(from_addr, to_addr, subject, content, files=[]):
    content_type = "plain"

    # 로그인 계정 이름(네이버 = 아이디, 구글 = 메일주소)
    username = "yijingue"

    # 비밀번호 (네이버 = 개인비밀번호, 애플리케이션 비밀번호, 구글 = 앱 비밀번호)
    password = "DLwlsrb1904#"

    # 구글 발송 서버 주소와 포트 (고정값)
    # smtp = "smtp.gmail.com"
    # port = 587

    # 네이버 발송 서버 주소와 포트 (고정값)
    smtp = "smtp.naver.com"
    port = 465
    # 메일 발송 정보를 저장하기 위한 객체
    msg = MIMEMultipart()

    msg['Subject'] = subject # 메일 제목
    msg['From'] = from_addr # 보내는 사람
    msg['To'] = to_addr # 받는 사람

    # 본문 설정 -> 메일의 내용과 형식 지정
    msg.attach(MIMEText(content, content_type))
    # 리스트 변수의 원소가 하나라도 존재할 경우 True
    if files:
        for f in files:
            # 바이너리(b) 형식으로 읽기(r) : 바이너리 읽기는 인코딩 설정 필요X
            with open(f, 'rb') as a_file:
                # 전체 경로에서 파일의 이름만 추출
                basename = os.path.basename(f)
                # 파일의 내용과 파일이름을 메일에 첨부할 형식으로 변환
                part = MIMEApplication(a_file.read(), Name = basename)

                # 파일첨부
                part['Content-Disposition'] = 'attachment; filename="%s"' % basename
                msg.attach(part)
    mail = SMTP(smtp)
    # 메일 서버 접속
    mail.ehlo()
    # 메일 서버 연동 설정
    mail.starttls()
    # 메일 서버 로그인
    mail.login(username, password)
    # 메일 보내기
    mail.sendmail(from_addr, to_addr, msg.as_string())
    # 메일 서버 접속 종료
    mail.quit()

# 모듈 테스트
if __name__ == "__main__":
    from_addr = 'yijingue@naver.com'
    to_addr = 'yijingue@gmail.com'
    subject = "메일 발송 테스트"
    content = """메일 발송 테스트입니다.
확인해봅시다."""
    files = ["X:\LJK\Python\Mega_IT\B.Python\hello.txt"]
    # 메일 발송 함수 테스트
    sendMail(from_addr, to_addr, subject, content, files)