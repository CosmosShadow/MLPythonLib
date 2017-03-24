# coding: utf-8
import multiprocessing as mp
import time

def foo_pool(x):
    time.sleep(2)
    return x*x


result_list = []
def log_result(result):
    result_list.append(result)

count = 10

def apply_async_with_callback():
    pool = mp.Pool()
    for i in range(count):
        pool.apply_async(foo_pool, args = (i, ), callback = log_result)
    pool.close()
    pool.join()
    # print(result_list)

def normal():
	result = []
	for i in range(count):
		result.append(foo_pool(i))

if __name__ == '__main__':
    # apply_async_with_callback()
    normal()









