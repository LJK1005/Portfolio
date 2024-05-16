class Protoss:
    name = None

    def __init__(self, name):
        self.name = name

    def move(self, target):
        print("[%s] %s로 이동" % (self.name, target))

    def attack(self, target):
        print("[%s] %s을(를) 공격" % (self.name, target))

    def niceMethod(self):
        print("팀원들은 절대 구현 못하는 겁나 어려운 코드")