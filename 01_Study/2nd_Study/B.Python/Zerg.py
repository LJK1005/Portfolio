from Protoss import Protoss
class Zerg(Protoss):
    def attack(self, target):
        print("%s가 %s에게 찌른다." % (self.name, target))