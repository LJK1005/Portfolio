from Protoss import Protoss
class Dragoon(Protoss):
    def attack(self, target):
        print("%s가 %s에게 쏜다." % (self.name, target))