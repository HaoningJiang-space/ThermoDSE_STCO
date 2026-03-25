
class mac_int8():
    def __init__(self):
        self.area = 457.1*1e-12
        self.cost = 0.4

class mac_fp16():
    def __init__(self):
        self.area = 1112.07*1e-12
        self.cost = 1.027

class sram_128KB():
    def __init__(self):
        self.area = 208338.64*1e-12
        self.rcost = 2.693
        self.wcost = 3.606

class sram_64KB():
    def __init__(self):
        self.area = 108068.39*1e-12
        self.rcost = 1.980
        self.wcost = 2.320

class sram_32KB():
    def __init__(self):
        self.area = 57933.26*1e-12
        self.rcost = 1.635
        self.wcost = 1.809

class regf_1KB():
    def __init__(self):
        self.area = 8607.76*1e-12
        self.rcost = 0.950
        self.wcost = 1.169
