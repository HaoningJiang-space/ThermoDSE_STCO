
class mac_int8():
    def __init__(self):
        self.area = 457.1*1e-12
        self.cost = 0.018

class mac_fp16():
    def __init__(self):
        self.area = 1112.07*1e-12
        self.cost = 0.0873

class sram_128KB():
    def __init__(self):
        self.area = 208338.64*1e-12
        self.rcost = 0.217125 * 8
        self.wcost = 0.234025 * 8

class sram_64KB():
    def __init__(self):
        self.area = 108068.39*1e-12
        self.rcost = 0.112 * 8
        self.wcost = 0.110675 * 8

class sram_32KB():
    def __init__(self):
        self.area = 57933.26*1e-12
        self.rcost =  0.106675 * 8
        self.wcost =  0.106675 * 8

class regf_1KB():
    def __init__(self):
        self.area = 8607.76*1e-12
        self.rcost = 0.049 * 8
        self.wcost = 0.064 * 8
