from collections import OrderedDict

class bufManager():
    def __init__(self, size):
        self.size = size
        self.stored = OrderedDict()
        self.occupy = 0

    def store_data(self, task_name, volume):
        if task_name in self.stored:
            raise ValueError(f'{task_name} already stored')
        elif self.occupy + volume > self.size:
            raise ValueError(f'{task_name} with volume {volume} will make buffer overflow')
        self.stored[task_name] = volume
        self.occupy += volume

    def free_space(self):
        return self.size - self.occupy

    def writeback_data(self):
        if len(self.stored) == 0:
            raise ValueError("buffer is empty. No data to write")
        wb_name, wb_volume = self.stored.popitem(last=True)    ## pop the first ofm in ubuf
        self.occupy -= wb_volume
        return wb_name

    def delete_data(self, task_name):
        if task_name not in self.stored:
            raise KeyError(f'{task_name} has not stored')
        volume = self.stored.pop(task_name)
        self.occupy -= volume

    def get_stored_list(self):
        return list(self.stored.keys())

    def get_stored_data(self, task_name):
        return self.stored[task_name]

    def copy(self):
        new_manager = bufManager(self.size)
        new_manager.stored = self.stored.copy()
        new_manager.occupy = self.occupy
        return new_manager

    def clear(self):
        self.stored.clear()

    def __str__(self):
        str_ = ''
        if len(self.stored) == 0:
            str_ += 'the buffer is empty'
        else:
            for key, value in self.stored.items():
                str_ += f'{key}: {value}\t'
            str_ += '\n'
        return str_
