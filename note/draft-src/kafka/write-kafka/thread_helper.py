import threading
from time import sleep

# class StatsWorker(threading.Thread):
#     def __init__(self,
#                  task="asr",
#                  model=None):
#         threading.Thread.__init__(self)
#         self._task = task
#         self._model = model

#     def run(self):
#         try:
#             sleep(1.0)
#             return 1
#         except Exception:
#             return 0

# t = StatsWorker()
# t.start()
# x = t.join()
# print(x)


def run():
    try:
        sleep(1.0)
        return 1
    except Exception:
        return 0

# 参考自: https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

# t = ThreadWithReturnValue(target=run, args=())

# t.start()
# sleep(0.9)
# x = t.join(0.01)
# sleep(0.3)
# alive_flag = t.is_alive()
# print(f"alive_flag: {alive_flag}")
# t.termi
# print(x)
