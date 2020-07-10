from collections import defaultdict
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import math
import random

class User:
    def __init__ (self, age, gender, salary, voting_intention):
        self.age = age 
        self.gender = gender
        self.salary = salary
        self.voting_intention = voting_intention

    def __str__ (self):
        return f'age : {self.age}, gender: {self.gender}, salary: {self.salary}, voting_intention: {self.voting_intention}'

    def __eq__ (self, other):
        return self.voting_intention == other.voting_intention
    
    def __hash__(self):
        return 1

def generates_base (n):
    l = []
    for i in range (n):
        age = random.randint(18, 35)
        gender = random.choice(['M', 'F'])
        salary = 1200 + random.random() * 1300
        voting_intention = random.choice(['Haddad', "Bolsonaro"])
        user = User(age, gender, salary, voting_intention);
        l.append(user)    
    return l

def highest_frequency_label_without_tie (users):
    frequencies = Counter (users)
    label, frequencie = frequencies.most_common(1)[0]
    amount_of_most_frequent = len([count for count in frequencies.values() if count == frequencie])

    if amount_of_most_frequent == 1:
        return label
    return highest_frequency_label_without_tie(users[:-1])

def distance (p1, p2):
    i = math.pow((p1.age - p2.age), 2)
    s = math.pow((1 if p1.gender == 'M' else 0) - (1 if p2.gender == 'M' else 0), 2)
    sal = math.pow((p1.salary - p2.salary), 2)
    return math.sqrt(i + s)

def knn (k, labeled_observations, new_observation):
    sorted_by_distance = sorted (labeled_observations, key= lambda obs: distance (obs, new_observation))
    k_nearest = sorted_by_distance[:k]
    result = highest_frequency_label_without_tie(k_nearest)
    return result.voting_intention

def base_size():
    return 100

def k_close_neighbors():
    return 5

def main():
    base = generates_base(base_size())
    for p in base:
        print(p)
    user = User(21, "F", 1700, None)
    user.voting_intention = knn(k_close_neighbors(), base, user)    
    print (user)

def cross_validation_leave_one_out():
    base = generates_base(base_size())
    chance_success = 0
    chance_error = 0
    
    for user in base:
        voting_intention = knn(k_close_neighbors(), base[:-1], user)
        if(voting_intention != user.voting_intention): 
            chance_error += + 1    
    print('Failures: ', chance_error)
    chance_success = ((len(base) - chance_error)/len(base))*base_size()
    chance_error = chance_error/len(base)*base_size()
    print(f'Chance Success: {chance_success}%')
    print(f'Chance Error: {chance_error}%')

def test_data_knn():
    main()
    cross_validation_leave_one_out()

test_data_knn()