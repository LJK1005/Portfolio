from Protoss import Protoss
class Archon(Protoss):
    def attack(self, target):
        print("%s가 %s에게 지진다." % (self.name, target))