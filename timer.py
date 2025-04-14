import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def print_time(self):
        current_time = time.time()
        interval = current_time - self.last_time
        self.last_time = current_time
<<<<<<< HEAD
        return inter
=======
        return interval
>>>>>>> 1835bf40df94672267435b0ddf57b48db2db6c03
