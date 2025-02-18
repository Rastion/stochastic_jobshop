import math
import sys
import random
from qubots.base_problem import BaseProblem
import os

def read_instance(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        lines = f.readlines()

    # The instance information is on the second line.
    first_line = lines[1].split()
    nb_jobs = int(first_line[0])
    nb_machines = int(first_line[1])
    nb_scenarios = int(first_line[2])

    # Read processing times for each job on each machine (given in processing order)
    # For each scenario s, for each job (lines 3 to 3+nb_jobs-1 in block s), read nb_machines integers.
    processing_times_in_processing_order_per_scenario = [
        [
            [int(lines[s * (nb_jobs + 1) + i].split()[j]) for j in range(nb_machines)]
            for i in range(3, 3 + nb_jobs)
        ]
        for s in range(nb_scenarios)
    ]

    # Read the machine order for each job.
    machine_order = [
        [int(x) - 1 for x in lines[4 + nb_scenarios * (nb_jobs + 1) + i].split()]
        for i in range(nb_jobs)
    ]

    # Reorder processing times so that:
    # processing_time[s][j][m] is the processing time of the task of job j that is processed on machine m in scenario s.
    processing_time_per_scenario = [
        [
            [processing_times_in_processing_order_per_scenario[s][j][machine_order[j].index(m)]
             for m in range(nb_machines)]
            for j in range(nb_jobs)
        ]
        for s in range(nb_scenarios)
    ]

    # Compute a trivial upper bound on start times.
    max_start = max(
        [sum(processing_time_per_scenario[s][j]) for s in range(nb_scenarios) for j in range(nb_jobs)]
    )
    return nb_jobs, nb_machines, nb_scenarios, processing_time_per_scenario, machine_order, max_start

class StochasticJobShopProblem(BaseProblem):
    """
    Stochastic Job Shop Scheduling Problem

    Each job j must be processed on every machine.
    For each job, the order in which machines are visited is given by machine_order[j] (a permutation of {0,…,nb_machines–1}).
    Processing times vary between scenarios.

    A candidate solution is represented as a list of length nb_machines;
    the element for machine m is a permutation of the job indices (0 … nb_jobs–1) indicating the order in which that machine processes its tasks.
    
    To evaluate a candidate solution, we simulate (in a simplified way) the schedule for each scenario as follows:
      - For each scenario s, we initialize each machine m’s available time to 0 and each job’s (overall) completion time to 0.
      - Then, for each machine m (in order m = 0,…,nb_machines–1), we process the jobs in the order given by candidate[m].  
        For each job j in candidate[m]:
          • Let op = machine_order[j].index(m) (i.e. the position of machine m in job j’s processing order).
          • The earliest start for job j on machine m is the completion time of its previous operation (or 0 if op = 0).
          • The actual start is the maximum of that value and the current availability of machine m.
          • The finish time is then the start plus processing_time_per_scenario[s][j][m].
          • We update the machine’s availability and record the job’s completion time.
      - The makespan for scenario s is the maximum completion time over all jobs.
    The objective is the maximum makespan over all scenarios.
    
    A penalty is added if, for any machine m, the candidate solution is not a permutation of {0,...,nb_jobs–1}.
    """
    def __init__(self, instance_file):
        self.instance_file = instance_file
        (self.nb_jobs,
         self.nb_machines,
         self.nb_scenarios,
         self.processing_time,
         self.machine_order,
         self.max_start) = read_instance(instance_file)

    def evaluate_solution(self, candidate) -> float:
        penalty = 0
        # Candidate is expected to be a list of length nb_machines,
        # where each element is a permutation (list) of job indices.
        for m in range(self.nb_machines):
            seq = candidate[m]
            if sorted(seq) != list(range(self.nb_jobs)):
                penalty += 1e6  # heavy penalty if not a valid permutation

        worst_makespan = 0
        # Simulate each scenario
        for s in range(self.nb_scenarios):
            # For each machine, initialize its available time.
            machine_time = [0] * self.nb_machines
            # For each job, record its last completion time (initially 0).
            job_completion = [0] * self.nb_jobs
            # Process machines in increasing order of machine index.
            # (Note: This is a simplified dispatching rule that uses the candidate orders.)
            for m in range(self.nb_machines):
                for j in candidate[m]:
                    # Determine the operation index for job j on machine m.
                    try:
                        op_index = self.machine_order[j].index(m)
                    except ValueError:
                        # Infeasible: machine m is not in the processing order of job j.
                        penalty += 1e6
                        op_index = 0
                    # Earliest start is the completion time of the previous operation (if any).
                    earliest = job_completion[j] if op_index > 0 else 0
                    start_time = max(machine_time[m], earliest)
                    finish_time = start_time + self.processing_time[s][j][m]
                    # Update the job's completion time and machine m's availability.
                    job_completion[j] = finish_time
                    machine_time[m] = finish_time
            scenario_makespan = max(job_completion)
            worst_makespan = max(worst_makespan, scenario_makespan)
        return worst_makespan + penalty

    def random_solution(self):
        """
        Generates a random candidate solution.
        For each machine, generate a random permutation of job indices.
        """
        candidate = []
        for m in range(self.nb_machines):
            perm = list(range(self.nb_jobs))
            random.shuffle(perm)
            candidate.append(perm)
        return candidate