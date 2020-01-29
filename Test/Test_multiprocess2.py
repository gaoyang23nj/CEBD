import multiprocessing
import time

class T1(object):
    def run(self):
        mgr = multiprocessing.Manager()
        d = mgr.dict()
        p = multiprocessing.Process(target=worker, args=(d, 12, 24))
        print('Start...')
        p.start()
        print('Join...')
        p.join()
        print('Results:')
        print(d)


def worker(d, key, value):
    print('=before==={}-{}====='.format(key, value))
    d[key] = value
    print(d)
    print('=====end====={}-{}============='.format(key, value))

if __name__ == '__main__':
    t1 = T1()
    t1.run()