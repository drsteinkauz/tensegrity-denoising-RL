import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def clip_time(self):
        current_time = time.time()
        interval = current_time - self.last_time
        self.last_time = current_time
        return interval

    def new_time_group(self,num):
        self.groups = num
        self.time_group = [0 for _ in range(num)]
        self.last_group = time.time()
        self.current_group = 0
    
    def clip_group(self):
        self.time_group[self.current_group]+=time.time()-self.last_group
        self.current_group = (self.current_group+1)%self.groups
        self.last_group = time.time()

    def group_print(self,tasks):
        try:
            for i,task in enumerate(tasks):
                print(task, " cost ", self.time_group[i], " seconds")
        except:
            pass