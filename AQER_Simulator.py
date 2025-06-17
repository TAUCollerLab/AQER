# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:11:13 2025

@author: anatl
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cosine 
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from typing import List, Tuple
import matplotlib.pyplot as plt


class Worker:
    def __init__(self, standard_deviation=0.0, bias=0.0):
        """
        Initializes a Worker object with standard deviation and bias.

        Parameters:
            standard_deviation (float): The standard deviation value (default is 0.0).
            bias (float): The bias value (default is 0.0).
        """
        self.standard_deviation = standard_deviation
        self.bias = bias

class Scenario:
    
    def __init__(self, num_questions, answer_dim):
        """
        Initializes a Scenario object with the specified number of questions and answer dimensions.

        Parameters:
            num_questions (int): The number of questions in the scenario.
            answer_dim (int): The dimension of each answer.
        """
        self._num_questions = num_questions
        self.answer_dim = answer_dim
        self.workers = []
        self.correct_answers = np.empty((0, 0))
        self.worker_answers = pd.DataFrame()
        
        
    @property
    def num_questions(self):
        return self._num_questions
    
    @num_questions.setter
    def num_questions(self, value: int):
        self._num_questions = value
        self.add_correct_answers()
        self.add_worker_answers()

    def add_workers(self, num_workers, standard_deviation, bias):
        """
        Adds a specified number of Worker objects to the scenario.

        Parameters:
            num_workers (int): The number of workers to add.
            standard_deviation (float): The standard deviation for the workers.
            bias (float): The bias for the workers.
        """
        for _ in range(num_workers):
            self.workers.append(Worker(standard_deviation, bias))

    def add_correct_answers(self, mean=0.0, std=1.0):
        """
        Generates and sets the correct answers for the scenario.

        Returns:
            numpy.ndarray: A numpy array of shape (num_questions, answer_dim) containing the correct answers.
        """
        # self.correct_answers = np.random.randn(self.num_questions, self.answer_dim)
        self.correct_answers = np.random.normal(loc=mean, scale=std, size=(self.num_questions, self.answer_dim))

        return self.correct_answers

    def add_worker_answers(self):
        """
        Generates worker answers based on the correct answers and worker properties.

        Returns:
            pandas.DataFrame: A DataFrame containing the worker answers for all questons in long format.
        """
        num_workers = len(self.workers)

        worker_answers = np.array([
            [
                np.random.normal(
                    loc=self.correct_answers[q] + w.bias, 
                    scale=w.standard_deviation , 
                    size=self.answer_dim
                ) for w in self.workers
            ] for q in range(self.num_questions)
        ])  # Shape: (num_questions, num_workers, answer_dim)

        # Reshape data into a long format for a DataFrame
        rows = []
        for q in range(self.num_questions):
            for w in range(num_workers):
                rows.append([q, w] + worker_answers[q, w].tolist())

        # Create DataFrame
        self.worker_answers = pd.DataFrame(rows, columns=["question", "worker"] + [f"x{i}" for i in range(self.answer_dim)])
        return self.worker_answers
    
    
class Scenario_by_correlation(Scenario):
    
    def __init__(self, num_questions, answer_dim, cor_between_weak_workers=0.5,weak_std=2):
        super().__init__(num_questions, answer_dim)
        self.cor_between_weak_workers = cor_between_weak_workers
        self.weak_std = weak_std    
    
    @staticmethod    
    def create_correlated_vector(v0, c1):
        n = len(v0)
        # Ensure v0 is mean-adjusted
        v0_adjusted = v0 - np.mean(v0)
        # Start with a scaled version of v0
        v1 = v0_adjusted * c1
        # Add noise to v1 to ensure it's not perfectly correlated or anticorrelated unless c1 is 1 or -1
        noise = np.random.normal(0, 1, n) * np.sqrt(1 - c1**2)
        v1 += noise
        # Adjust the mean and scale of v1 to fine-tune the correlation
        # This step is technically not necessary for correlation, but can be used for other purposes
        v1 = (v1 - np.mean(v1)) / np.std(v1)
        return v1
        
    
    def add_worker_answers(self): # Overide
        rows = []
        for q in range(self.num_questions):
            is_first_worker = True
            for idx, w in enumerate(self.workers):
                if w.standard_deviation >= self.weak_std:
                    if is_first_worker:
                        is_first_worker = False
                        # generating a weak worker answer (only once per question)
                        first_weak_ans = np.random.normal(loc=self.correct_answers[q] + w.bias, scale=w.standard_deviation, size=self.answer_dim
                        )
                    answer = self.create_correlated_vector(first_weak_ans, self.cor_between_weak_workers)
                else:
                    answer = np.random.normal(loc=self.correct_answers[q] + w.bias, scale=w.standard_deviation, size=self.answer_dim)
                # Use the worker index (idx) instead of the Worker object
                rows.append([q, idx] + answer.tolist())
        self.worker_answers = pd.DataFrame(
            rows, 
            columns=["question", "worker"] + [f"x{i}" for i in range(self.answer_dim)]
        )
        return self.worker_answers

    

class Simulator:
    def __init__(self, max_iterations, threshold, simulation_repetitions):
        """
        Initializes a Simulator object with a specified maximum number of iterations and threshold.

        Parameters:
            max_iterations (int): The maximum number of iterations for simulations for EM.
            threshold (float): The threshold for convergence.
            simulation_repetitions (int): The number of simulation repetitions 
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.simulation_repetitions = simulation_repetitions

    @staticmethod
    def weightedAvgAnswer(grp, workers_skill_level):
        """
        Computes the weighted average of answers based on workers' skill levels.

        Parameters:
            grp (numpy.ndarray): The group of answers.
            workers_skill_level (numpy.ndarray): The skill levels of workers.

        Returns:
            numpy.ndarray: The weighted average answer.
        """
        return np.average(grp, axis=0, weights=workers_skill_level)

    def grades_expectation_maximization(self, answers, max_iter, is_log=False,
                                        print_data=False, log_file=None, sim_repetition=0):
        """
        Implements the Grades Expectation-Maximization algorithm.

        Parameters:
            answers (pd.DataFrame): The answers DataFrame.
            is_log (bool): Whether to log the process.
            print_data (bool): Whether to print debug information.
            log_file (str): File path for logging data.
            sim_repetition (int): Simulation repetition number.

        Returns:
            tuple: Workers' grade array and skill level array.
        """
        answers = answers.sort_values(by=['worker', 'question'])
        worker_ids = sorted(pd.unique(answers['worker']))
        questions = sorted(pd.unique(answers['question']))

        log = []
        workers_skill_level = np.full(len(worker_ids), 1 / len(worker_ids))

        for e in range(max_iter):
            prev_workers_skill_level = workers_skill_level
            avg_answers_per_quest = pd.DataFrame()
            for t in range(len(questions)):
                relevant_rows = answers[answers['question'] == questions[t]]
                relevant_rows = relevant_rows.drop(columns=['question', 'worker'])
                weighted_array = self.weightedAvgAnswer(relevant_rows.to_numpy(), workers_skill_level)
                avg_answers_per_quest = pd.concat([avg_answers_per_quest, pd.DataFrame(weighted_array).T], ignore_index=True, axis=0)
            avg_answers_per_quest = avg_answers_per_quest.set_index(pd.Series(questions))

            workers_grade_array = []
            for ID in worker_ids:
                answers_by_worker = answers[answers['worker'] == ID]
                grade = 0
                curr_workers_ans = answers_by_worker.iloc[:, 2:].to_numpy()
                for j in range(len(avg_answers_per_quest)):
                    # cur_grade = (2 - cosine(curr_workers_ans[j], avg_answers_per_quest.to_numpy()[j]) / 2)
                    cur_grade = cosine_similarity([curr_workers_ans[j]], [avg_answers_per_quest.to_numpy()[j]])[0, 0]
                    grade += cur_grade
                    if is_log:
                        log.append({
                            "epoch": e, "worker": ID, "question": j,
                            "answer": curr_workers_ans[j],
                            "avg_ans": np.round(avg_answers_per_quest.to_numpy()[j], 2),
                            "grade": cur_grade, "sim_repetition": sim_repetition
                        })
                workers_grade_array.append(grade / len(avg_answers_per_quest))
            workers_grade_array = minmax_scale(workers_grade_array, feature_range=(0, 1))

            grade_sum = sum(map(abs, workers_grade_array))
            workers_skill_level = [g / grade_sum for g in workers_grade_array]

            if mean_squared_error(prev_workers_skill_level, workers_skill_level, squared=False) < self.threshold:
                break

        if is_log:
            log_df = pd.DataFrame(log)
            if log_file is None:
                log_file = "log_" + str(datetime.now().strftime("%Y-%m-%d_%H_%M_%S")) + ".csv"
            log_df.to_csv(log_file, index=False)

        return workers_grade_array, workers_skill_level



    def run_simulation(self, scenario: Scenario) -> Tuple[float, float, float, float]:
        """
        Runs the simulation multiple times and calculates the average and standard deviation
        of performance_epoch_0 and performance_with_iter.
    
        Parameters:
            scenario (Scenario): The scenario to simulate.
    
        Returns:
            tuple: Averages and standard deviations of performance_epoch_0 and performance_with_iter.
        """
        performance_epoch_0_list = []
        performance_with_iter_list = []
    
        for _ in range(self.simulation_repetitions):
            # Generate fresh answers for each repetition
            scenario.add_correct_answers()
            scenario.add_worker_answers()
    
            # Grades after 1 epoch
            worker_grades_0, workers_skill_level_0 = self.grades_expectation_maximization(
                                                            scenario.worker_answers,max_iter= 1, is_log=False
                
            )
    
            # Grades after multiple epochs
            worker_grades, workers_skill_level = self.grades_expectation_maximization(
                                                            scenario.worker_answers,self.max_iterations, is_log=False
            )
    
            # Calculate workers' actual grades
            actual_grades = []
            correct_ans_df = pd.DataFrame(scenario.correct_answers)
            for w in range(len(scenario.workers)):
                worker_answers_array = scenario.worker_answers[scenario.worker_answers['worker'] == w].iloc[:, 2:].to_numpy()
                correct_answers_array = correct_ans_df.to_numpy()
                similarities = cosine_similarity(worker_answers_array, correct_answers_array)
                actual_grades.append(np.mean(similarities))
    
            performance_epoch_0 = np.corrcoef(actual_grades, worker_grades_0)[0, 1]
            performance_with_iter = np.corrcoef(actual_grades, worker_grades)[0, 1]
    
            performance_epoch_0_list.append(performance_epoch_0)
            performance_with_iter_list.append(performance_with_iter)
    
        # Calculate averages and standard deviations
        avg_epoch_0 = np.mean(performance_epoch_0_list)
        std_epoch_0 = np.std(performance_epoch_0_list)
        avg_with_iter = np.mean(performance_with_iter_list)
        std_with_iter = np.std(performance_with_iter_list)
    
        return avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter


# Usage Example

if __name__=="__main__":
    # create scenario:
    scenario = Scenario(num_questions=15, answer_dim=100)
    scenario.add_workers(num_workers=20, standard_deviation=2.0, bias=0)
    scenario.add_workers(num_workers=20, standard_deviation=0.5, bias=0)
    scenario.add_workers(num_workers=20, standard_deviation=1, bias=0)

    scenario.add_correct_answers()
    scenario.add_worker_answers()
    
    # #run simulation of this scenario
    simulator = Simulator(max_iterations=1000, threshold=0.00001, simulation_repetitions=100)
    avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter = simulator.run_simulation(scenario)
    print(avg_epoch_0, std_epoch_0, avg_with_iter, std_with_iter)
     
