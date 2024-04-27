# 객체지향 언어
#   은닉성 : 프로퍼티
#   상속성 : extends
#     -> 다형성, 추상화

# 기본 파이썬에서 F9를 사용하면 중단점을 지정할 수 있음
# F5로 실행시 해당 지점 이전까지만 실행함
# 멈춘 지점에서 F11를 누르면 코드가 1줄씩 실행됨 (디버깅 기능)
# F5를 다시 누르면 나머지를 자동으로 실행함

# 클래스 상속
# 기존에 만들어둔 클래스의 변수, 함수 등을 다른 클래스 생성 시에 그대로 물려받는 기능
# class 새 클래스(기존 클래스):
#   ~~~~~
# 코드 없이 블록만 넣을 경우 pass만 넣으면 됨

# 상속 기능의 확장 : 부모 class에서 전달받은 변수와 함수 외에 다른 요소를 추가
# 상속 기능의 변경 : 부모 class에서 전달받은 변수와 함수를 변경, 전달받은 함수나 변수와 겹치는 이름가진 요소는 자식에서 설정한 값으로 덮어씌워짐 (메소드 오버라이드)

# @abstractmethod (abc에서 * import) : 상속받은 부모클래스가 추상화되면 자식 클래스는 전달받은 모든 요소를 오버라이드를 반드시 시켜야 한다
# 따라서 상속받는 자식의 코드 수정 오류로 근본적인 오류가 발생했을 떄 이상이 없는 부모 클래스에 추상화를 걸어 어느 자식 클래스에서 코드가 잘못되었는지 파악 가능
# ex) 자식클래스에서 오버라이드하는 함수명이 잘못되었을 경우 부모가 추상화되면 자식클래스는 해당 함수명을 오버라이드 하지 못하므로 오류가 감지됨


# %%

class Hello:
    name = None
    
    def __init__(self, name):
        self.name = name

    def say(self):
        print(f"안녕하세요. {self.name}입니다.")

class Hello2(Hello):
    pass

h = Hello2("철수")
h.say()

# %%

from Zerg import Zerg
from Dragoon import Dragoon
from Archon import Archon

if __name__ == "__main__":
    z = Zerg("질럿1호")
    d = Dragoon("드라군1호")
    a = Archon("아칸1호")

    z.move("적의본진")
    d.move("적의본진")
    a.move("적의본진")

    z.attack("적의본진")
    d.attack("적의본진")
    a.attack("적의본진")


# %%

print("안녕하세요")
print("시작합니다")

for i in range(0, 10):
    str = ""
    for j in range(0, i+1):
        str += "*"

    print(str)

# %%