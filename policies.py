"""
This file contains the pARP, bBRP, and pMBRP policies as described by Schouten et. al (2022). Each class has an
__init__() function to create the policy and a policy() function that takes in a state and outputs the number of PMs
for that given state. It calculates this by applying the policy to each wind turbine individually. For more information
on these policies, see the paper section 4.2.
"""

class ARP:
    def __init__(self, critical_maintenance_age: list, name = "ARP"):
        self.name = name
        self.cma = critical_maintenance_age

    def policy(self, state: list):
        period_age = self.cma[state[0] % len(self.cma)]
        pm = 0
        for age in state[1:]:
            if age >= period_age: pm += 1
        return pm


class BRP:
    def __init__(self, intervals: list):
        self.name = "BRP"
        self.intervals = intervals
        self.interval_index = 0
        self.time_till_pm = self.intervals[self.interval_index]

    def update_interval(self):
        if self.time_till_pm == 0:
            if self.interval_index < len(self.intervals) - 1: self.interval_index += 1
            if self.interval_index == len(self.intervals) - 1: self.interval_index = 0
            self.time_till_pm = self.intervals[self.interval_index]

    def policy(self, state: list):
        pm = 0
        for age in state[1:]:
            if age == 0: cm += 1

        if self.time_till_pm == 0:
            pm = len(state) - 1
        self.update_interval()
        return pm


class MBRP:
    def __init__(self, critical_maintenance_age: list, intervals: list):
        self.name = "MBRP"
        self.intervals = intervals
        self.cma = critical_maintenance_age
        self.interval_index = 0
        self.time_till_pm = self.intervals[self.interval_index]

    def update_interval(self):
        if self.time_till_pm == 0:
            if self.interval_index < len(self.intervals) - 1: self.interval_index += 1
            if self.interval_index == len(self.intervals) - 1: self.interval_index = 0
            self.time_till_pm = self.intervals[self.interval_index]

    def policy(self, state: list):
        pm = 0
        t = state[0]
        for age in state[1:]:
            if self.time_till_pm == 0 and age >= self.cma[t]: pm += 1
        self.update_interval()
        return pm
