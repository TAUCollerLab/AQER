class AQER:
    def __init__(self, answers_embeddings max_iterations, threshold):
        """
        Initializes an AQER object with a specified maximum number of iterations and threshold.

        Parameters:
            answers_embeddings (pd.DataFrame): The embeddings of answers provided by workers. Columns format: ["question",	"worker", "x1",...,"xn"]
                                               Where "question" is the question's id, "worker" is the worker's id, "x1"..."xn" are the answer's embedding dimensions 
            max_iterations (int): The maximum number of iterations for simulations for EM.
            threshold (float): The threshold for convergence.
            
        """
        self.max_iterations = max_iterations
        self.threshold = threshold

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

    def grades_expectation_maximization(self, max_iter, is_log=False,
                                        print_data=False, log_file=None):
        """
        Implements the Grades Expectation-Maximization algorithm.

        Parameters:
            is_log (bool): Whether to log the process.
            print_data (bool): Whether to print debug information.
            log_file (str): File path for logging data.
            
        Returns:
            tuple: Workers' grade array and skill level array.
        """
        answers = self.answers.sort_values(by=['worker', 'question'])
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
                            "grade": cur_grade
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



    
