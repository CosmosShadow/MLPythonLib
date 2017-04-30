# coding: utf-8
import time
import uuid
import random
import numpy as np
import pandas as pd
import lake

question_count = 30000
count_to_query = 100

print '总题数: %d   获取难度数: %d' % (question_count, count_to_query)

questions = [str(uuid.uuid4()) for _ in range(question_count)]
d = list(np.random.random(question_count))
questions_to_query = random.sample(questions, count_to_query)


@lake.decorator.time_cost_red
def get_pandas_d(questions, d):
	return pd.DataFrame(np.array(d), index=questions)


@lake.decorator.time_cost_red
def get_dict_d(questions, d):
	return dict(zip(questions, d))


@lake.decorator.time_cost_red

def qurey_pandas_d(pd_d, questions_to_query):
	return pd_d.loc[questions_to_query].values


@lake.decorator.time_cost_red
def qurey_dict_d(dict_d, questions_to_query):
	return [dict_d[item] for item in questions_to_query]


pd_d = get_pandas_d(questions, d)
dict_d = get_dict_d(questions, d)

qurey_pandas_d(pd_d, questions_to_query)
qurey_dict_d(dict_d, questions_to_query)

