from redis import Redis
from rq import Queue
import pickle
import numpy as np
import pandas as pd


# rediscon = Redis('192.168.1.253', 6379, password=None)
rediscon = Redis()
redisQ = Queue('SLAM',connection=rediscon)


# job = q.enqueue(
#              stktechret.count_words_at_url, 'http://nvie.com')
# job = q.enqueue(
#              stktechret.count_words_at_url, 'http://nvie.com')
# job = q.enqueue(
#              stktechret.count_words_at_url, 'http://nvie.com')
# print("fff ",q.empty())
# print(job.result)
# time.sleep(2)
# print(job.result)   # => 889
# print("fff ",q.empty())

# df = pd.DataFrame({'A':np.random.rand(100000),'B':np.random.rand(100000)})

# rediscon.hset("Users","Name",pkl.dumps(df, protocol=pkl.HIGHEST_PROTOCOL))

# dfstr = rediscon.hget("Users","Name")
# dfrecon = pkl.loads(dfstr)
# dfrecon.head()

# while True:
#     D=[]
#     flg = True
#     for jb in jobs:
#         if jb.result is None:
#             flg=False
#             break
#         else:
#             D.append(jb.result)
#     if flg is True:
#         break
#     time.sleep(5)
#     print("polling")

class DD:
    def __init__(self,states):
        self.states = states
        self.A = np.random.rand(len(states))
    
    def __call__(self,substates):
        print(self.states)
        print(substates)
        a = np.sum(self.A[substates])
        print(a)
        return a

d= DD(np.array([1,2,3,4,5,6]))
def putme():
    job = redisQ.enqueue(d, np.array([2,3,4]))

class Cache2redis:
    def __init__(folder=None):
        pass