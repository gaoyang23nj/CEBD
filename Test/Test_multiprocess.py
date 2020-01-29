import multiprocessing
import time


def worker(d, key, value):
    print('=before==={}-{}====='.format(key, value))
    d[key] = value
    print(d)
    print('=====end====={}-{}============='.format(key, value))


# if __name__ == '__main__':
#     mgr = multiprocessing.Manager()
#     d = mgr.dict()
#     jobs = [multiprocessing.Process(target=worker, args=(d, i, i * 2))
#             for i in range(10)
#             ]
#     for j in jobs:
#         j.start()
#     for j in jobs:
#         j.join()
#     print('Results:')
#     for key, value in enumerate(dict(d)):
#         print("%s=%s" % (key, value))

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    jobs = []
    for i in range(10):
        j = multiprocessing.Process(target=worker, args=(d, i, i * 2))
        jobs.append(j)
    for j in jobs:
        print('Start...')
        j.start()
    for j in jobs:
        print('Join...')
        j.join()
    print('Results:')
    for key, value in enumerate(dict(d)):
        print("%s=%s" % (key, value))
    print(d)
    for i in d:  # 遍历字典中的键
        print('{}={}'.format(i, d[i]))