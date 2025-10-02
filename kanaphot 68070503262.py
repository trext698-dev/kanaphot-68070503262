Import Car class from car.py

define car1 with Car('Toyota',1000)

run the method like display(), rent(), return_car()

from car import Car
car1 = Car('Toyota',1000)
car1.display()
car1.rent(1)
car1.return_car()
Model: Toyota
Rent per day: ฿1000
Availability: Available
Car rented. Total cost = ฿1000
Car returned
Example 1 – Function: Compute Average Load on a Beam Write a Python function average_load(loads) that receives a NumPy array of loads (kN) on different sections of a beam and returns the average load.

[12.5, 13.0, 12.0, 11.5, 12.8]

import numpy as np

def average_load(loads):
    return np.mean(loads)

beam_loads = np.array([12.5, 13.0, 12.0, 11.5, 12.8])
average_load = average_load(beam_loads)
print(average_load)
Example 2 – Function: Create a Diagonal Stiffness Matrix

Question: Write a function stiffness_matrix(values) that takes a 1D NumPy array of stiffness values and returns a diagonal matrix with those values on the diagonal.

import numpy as np

def stiffness_matrix(values):
    return np.diag(values)

stiffness_values = np.array([10, 20, 30])
stiffness_matrix_result = stiffness_matrix(stiffness_values)
print("Stiffness Matrix:")
print(stiffness_matrix_result)
Stiffness Matrix:
[[10  0  0]
 [ 0 20  0]
 [ 0  0 30]]
Example 3 – Class: ConcreteStrength

Question: Create a class ConcreteStrength to store compressive strength test data.

The class should have:

an attribute samples (NumPy array of strengths)

a method mean_strength() returning the mean

a method std_strength() returning the standard deviation

import numpy as np

class ConcreteStrength:
  def __init__(self, samples):
      
      self.samples = np.array(samples)

  def mean_strength(self):
     
     return np.mean(self.samples)
  
  def std_strength(self):

    return np.std(self.samples)

strength_data = ConcreteStrength([25.4, 26.1, 24.8, 25.9, 26.3])
mean_value = strength_data.mean_strength
print(mean_value)
stp_value = strength_data.std_strength()
print(stp_value)
<bound method ConcreteStrength.mean_strength of <__main__.ConcreteStrength object at 0x00000283ABD3D940>>
0.540370243444252
Example 4 – Class: GridElevation

Question: Write a class GridElevation representing a 2D grid of elevations. It should have:

elevations (NumPy 2D array)

max_elevation() method

min_elevation() method

profile(row) method returning one row’s elevation profile as a 1D NumPy array

import numpy as np

class GridElevation:
  def __init__(self, elevations):

     self.elevations = np.array(elevations)

  def max_elevation(self):

    return np.max(self.elevations)

  def min_elevation(self):

    return np.min(self.elevations)

  def profile(self, row):

    return self.elevations[row]

grid = GridElevation([[5.2,5.4,5.3],
                      [5.1,5.3,5.5],
                      [5.0,5.2,5.4]])

max_value = grid.max_elevation()
print(max_value)
min_value = grid.min_elevation()
print(min_value)
profile_value = grid.profile(0)
print(profile_value)
5.5
5.0
[5.2 5.4 5.3]
FitnessTracker Exercises

You are given 7 days of step counts for 3 people (rows = days, columns = people):

Create a class called FitnessTracker that:

Accepts the step data as a NumPy array.

Optionally accepts a list of names for the people.

Stores the data as an attribute.

Add the following methods to your class:

total_steps_per_person() → Returns total steps for each person.

total_steps_per_day() → Returns total steps for each day.

average_per_person() → Returns average steps per person.

best_day_overall() → Returns the day (1-based) with the highest average steps across all people.

person_max_on_day(day_index) → Returns the person who walked the most on a given day (0-based index).

days_over_threshold(threshold) → Returns the number of days each person exceeded a given threshold.

percent_change_day(day1, day2) → Returns the percentage change in steps from day1 to day2 for each person.

Add a method normalize_per_person() that:

Returns a Min-Max normalized array per person.

For each person (column), scale their steps to the range [0, 1] using:

Min-Max normalization: 
 

Each column should be normalized independently based on that person’s min and max steps.

import numpy as np

class FitnessTracker:
  def __init__(self, steps, names=None):
    self.steps = np.array(steps)
    self.days, self.people = self.steps.shape
    self.names = names

  def total_steps_per_person(self):
    return np.sum(self.steps,axis=0)

  def total_steps_per_day(self):
    return np.sum(self.steps,axis=1)

  def average_per_person(self):
    return np.mean(self.steps,axis=0)

  def best_day_overall(self):
    day_avg= (np.mean(self.steps,axis=1))
    return np.argmax(day_avg)+1

  def person_max_on_day(self, day_index):
    index = np.argmax(self.steps[day_index, :])
    return self.names[index]

  def days_over_threshold(self, threshold=10000):
    return np.sum(self.steps> threshold, axis=0)

  def percent_change_day(self, day1, day2):
    return (self.steps[day2, :] - self.steps[day1,:]) / self.steps[day1,:]*100


  def minmax_normalize_per_person(self):
    min_values = np.min(self.steps, axis=0)
    max_values = np.max(self.steps, axis=0)
    normalized = (self.steps - min_values) / (max_values-min_values)
    return normalized

steps = np.array([[8000, 10000, 9500],
                  [9000, 11000, 8700],
                  [7500,  9800, 10200],
                  [8200, 10500, 9700],
                  [8800, 11500, 9400],
                  [9100, 10800, 10100],
                  [8500,  9900, 9800]])

tracker = FitnessTracker(steps, names=['Alice', 'Bob', 'Charlie'])

print("Total steps per person:", tracker.total_steps_per_person())
print("Total steps per day:", tracker.total_steps_per_day())
print("Average per person:", tracker.average_per_person())
print("Best day overall (highest avg):", tracker.best_day_overall())
print("Who walked the most on Day 5?", tracker.person_max_on_day(4))
print("Days each person >10k:", tracker.days_over_threshold(10000))
print("Percent change Day 1→Day 7:", tracker.percent_change_day(0,6))
print("Min-Max Normalized steps:\n", tracker.minmax_normalize_per_person())
Total steps per person: [59100 73500 67400]
Total steps per day: [27500 28700 27500 28400 29700 30000 28200]
Average per person: [ 8442.85714286 10500.          9628.57142857]
Best day overall (highest avg): 6
Who walked the most on Day 5? Bob
Days each person >10k: [0 4 2]
Percent change Day 1→Day 7: [ 6.25       -1.          3.15789474]
Min-Max Normalized steps:
 [[0.3125     0.11764706 0.53333333]
 [0.9375     0.70588235 0.        ]
 [0.         0.         1.        ]
 [0.4375     0.41176471 0.66666667]
 [0.8125     1.         0.46666667]
 [1.         0.58823529 0.93333333]
 [0.625      0.05882353 0.73333333]]
