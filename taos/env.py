"""
Env setting and generation.
"""
import math
from functools import reduce
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Env parameter settings.
# NOTE: Some of them are discarded when using real job trace.
# ----------------------------------------------------------------------------

# HINT: `num_tasks` should be large to minimize the affect of intra-job co-execution
MIN_TASK_NUM_IN_TG = 1
MAX_TASK_NUM_IN_TG = 200

MIN_TG_NUM = 1
MAX_TG_NUM = 5

JOB_NUM = 250
SITE_NUM = 100

MAX_ARRIVAL_TU = 260  # HINT: This should >= JOB_NUM

# HINT: System utilization is used to tune the inter-job arrivals
UTILIZATION_FACTOR = 0.75

# The parameter for Zipf distribution
ALPHA = 2

# Setting SCALE_INTER_ARRIVALS \to 0 to make a fierce contention
SCALE_INTER_ARRIVALS = 0.005

MIN_AS_NUM_FOR_TG = 8
# HINT: A larger `MAX_AS_NUM_FOR_TG` leads to a better performance of SJF algorithms
MAX_AS_NUM_FOR_TG = 12  # HINT: This should <= SITE_NUM

# Parameters for generating the computing capacities
MIN_MU = 3
# HINT: `MAX_MU` should be small to minimize the affect of intra-job co-execution.
#  However, a smaller it will lead to a larger computation overhead of order scheduling
MAX_MU = 5

# In heterogenous mode, the tasks of any job can have different running times
HOMOGENEOUS_MODE = "HOMOGENEOUS_MODE"
HETEROGENOUS_MODE = "HETEROGENOUS_MODE"
MODE = HETEROGENOUS_MODE

# Parameters for long-tail distribution in the heterogenous mode
RUNNING_TIME_MU = 1
RUNNING_TIME_SIGMA = 0.5

# ----------------------------------------------------------------------------
# Where are the generated info. stored.
# ----------------------------------------------------------------------------
jobs = []
sites = []
NUM_TASKS = 0


class Job(object):
    def __init__(self, index, task_groups, arrival_time):
        self.index = index
        self.task_groups = task_groups
        self.arrival_time = arrival_time
        self.available_sites = None  # The set S_g for this job J_g

        self.completion_time = None

    def set_available_sites(self):
        """
        Calculate S_g (the join of the AS sets of each task group).
        """
        self.available_sites = reduce(lambda x, y: x | y,
                                      [tg.available_sites for tg in self.task_groups])

    def reset(self):
        self.completion_time = None

    def _collect_task_group_info(self):
        info = ""
        for tg in self.task_groups:
            info += "\t" + str(tg) + "\n"
        return info

    def __str__(self):
        return "job index: {0}\nnum. of task groups: {1}, arrival time: {2}, S_g: {3}\n" \
               "task group info:\n{4}".format(self.index,
                                              len(self.task_groups),
                                              self.arrival_time,
                                              {site.index for site in self.available_sites},
                                              self._collect_task_group_info())


class TaskGroup(object):
    def __init__(self, index, job, num_tasks, available_sites):
        self.index = index
        self.job = job
        self.num_tasks = num_tasks
        self.num_unfinished_tasks = num_tasks

        self.available_sites = available_sites  # The set S_g^k for this task group T_g^k

        # Real execution time of each task in this task group.
        # In Alibaba trace, we use the instance duration as the task's running time.
        # All instances of the same task group has the same execution time
        self.real_running_time = None

    def reset(self):
        self.num_unfinished_tasks = self.num_tasks

    def __str__(self):
        return "index: {0}, num. of tasks: {1}, ASs: {2}, Each task's running time: {3}".format(
            self.index, self.num_tasks, {site.index for site in self.available_sites},
            self.real_running_time
        )


class Site(object):
    def __init__(self, index, capacities):
        self.index = index
        self.capacities = capacities  # A list of \mu_m^g for each job on this site S_m

        # The estimated backlog size of this site.
        # Here we use the word "estimated" because it is the number of allocated time units,
        # not the actual time duration for running tasks
        self.estimated_bklg_size = 0

        # The actual time duration for running tasks
        self.true_bklg_size = 0

    def __str__(self):
        return "index: {0}, capacities: {1}".format(
            self.index, [cap for cap in self.capacities]
        )

    def reset(self):
        self.estimated_bklg_size = 0


def create_env():
    """
    Synthesize a scheduling environment.
    """
    # Generate each job's non-repeat arrival time
    arrivals = np.arange(MAX_ARRIVAL_TU)
    np.random.shuffle(arrivals)

    # Generate the number of task groups for each job
    num_task_groups = np.random.randint(MIN_TG_NUM, MAX_TG_NUM + 1, size=JOB_NUM)

    # Generate sites
    # If a site is not available to some job, the corresponding \mu should be zero. We just skip it here.
    # In real scenarios, you should set \mu based on the resource request and equipment.
    capacities = np.random.randint(MIN_MU, MAX_MU + 1, size=(SITE_NUM, JOB_NUM))
    for site_idx in range(SITE_NUM):
        site = Site(site_idx, capacities=capacities[site_idx])
        sites.append(site)

    for job_idx in range(JOB_NUM):
        job = Job(job_idx, task_groups=None, arrival_time=arrivals[job_idx])

        # Generate the number of tasks for this task group
        num_tasks = np.random.randint(MIN_TASK_NUM_IN_TG, MAX_TASK_NUM_IN_TG + 1, size=num_task_groups[job_idx])
        # Generate AS set for this task group
        num_available_sites = np.random.randint(MIN_AS_NUM_FOR_TG, MAX_AS_NUM_FOR_TG + 1, size=num_task_groups[job_idx])

        tgs = []
        for tg_idx in range(num_task_groups[job_idx]):
            shuffled_sites = np.arange(SITE_NUM)
            np.random.shuffle(shuffled_sites)
            as_set = shuffled_sites[:num_available_sites[tg_idx]]
            available_sites = {site for site in sites if site.index in as_set}

            tg = TaskGroup(tg_idx, job, num_tasks=num_tasks[tg_idx], available_sites=available_sites)

            global MODE
            if MODE == HETEROGENOUS_MODE:
                # Generate each task's real running time with a long-tail distribution
                tg.real_running_time = int(np.ceil(np.random.lognormal(mean=RUNNING_TIME_MU,
                                                                       sigma=RUNNING_TIME_SIGMA,
                                                                       size=1)))

            elif MODE == HOMOGENEOUS_MODE:
                # Set all task's running time as 1
                tg.real_running_time = 1

            tgs.append(tg)

        job.task_groups = tgs
        job.set_available_sites()

        jobs.append(job)

    global NUM_TASKS
    for job in jobs:
        NUM_TASKS += sum(tg.num_tasks for tg in job.task_groups)


def from_trace():
    """
    Read from trace. In this data, a job contains multiple tasks, different tasks executes
    different computing logics. Instance is the smallest scheduling unit of batch workload,
    and all instances within a task execute exactly the same binary with the same resource
    request, but with different input data. Thus, `task' in this data corresponds to task
    group in our model, and `instance' corresponds to task.

    NOTE:
        1. We filter out large task groups (#tasks > 500) since these task groups will be
        outstanding significantly, which makes the comparison less obvious.
        2. Task durations and job arrivals are derived from the events record
    """
    # 1. Read from trace
    df = pd.read_table("../trace/batch_task.csv")

    data = {}  # Save the task group info
    arrivals = {}  # Save the job arrival times
    job_index = -1
    tg_index = 0

    id_list = []
    for line in df.values:
        line = line[0].split(",")

        # The three data segments are needed
        num_tasks = int(line[4])
        # Filter out large task groups
        if num_tasks > 500:
            continue
        job_id = int(line[2])
        tg_duration = int(math.ceil((int(line[1]) - int(line[0])) / 100))
        if tg_duration <= 0:
            tg_duration = 1
        elif tg_duration > 100:
            tg_duration = 100

        if job_id not in id_list:
            if len(id_list) >= JOB_NUM:
                break

            id_list.append(job_id)
            job_index += 1
            data[job_index] = {}
            arrivals[job_index] = int(line[0])
            tg_index = 0

        data[job_index][tg_index] = (num_tasks, tg_duration)
        tg_index += 1

    # aver_num_tg = 0
    # for v in data.values():
    #     aver_num_tg += len(v)
    # print(aver_num_tg/JOB_NUM)

    global NUM_TASKS
    NUM_TASKS = 0
    for tg_info in data.values():
        for v in tg_info.values():
            NUM_TASKS += v[0]

    # 2. Class instance generation
    # Scale the arrival intervals with utilization factor
    # arrivals = np.arange(MAX_ARRIVAL_TU)
    # np.random.shuffle(arrivals)
    min_arrival = min(a for a in arrivals.values())
    for job_index in arrivals.keys():
        arrivals[job_index] -= min_arrival
        arrivals[job_index] = int(np.ceil(SCALE_INTER_ARRIVALS * arrivals[job_index] / UTILIZATION_FACTOR))

    # Collect the number of task groups for each job
    num_task_groups = [len(v) for v in data.values()]

    # Generate sites
    # If a site is not available to some job, the corresponding \mu should be zero. We just skip it here.
    # In real scenarios, you should set \mu based on the resource request and equipment.
    capacities = np.random.randint(MIN_MU, MAX_MU + 1, size=(SITE_NUM, JOB_NUM))
    for site_idx in range(SITE_NUM):
        site = Site(site_idx, capacities=capacities[site_idx])
        sites.append(site)

    # Generate available sites with Zipf distribution
    shuffled_sites = np.arange(SITE_NUM)
    np.random.shuffle(shuffled_sites)
    probs = np.power([1 / (i + 1) for i in shuffled_sites], ALPHA)
    if ALPHA == 0:
        probs = probs / SITE_NUM
    else:
        probs = probs / sum(probs)

    for job_idx in range(JOB_NUM):
        job = Job(job_idx, task_groups=None, arrival_time=arrivals[job_idx])

        # Generate the number of tasks for each task group
        num_tasks = []
        task_durations = []
        for tg_info in data[job_idx].values():
            num_tasks.append(tg_info[0])
            task_durations.append(tg_info[1])

        # Generate AS set for this task group
        num_available_sites = np.random.randint(MIN_AS_NUM_FOR_TG, MAX_AS_NUM_FOR_TG + 1, size=num_task_groups[job_idx])

        tgs = []
        for tg_idx in range(num_task_groups[job_idx]):
            start_site_idx = np.random.choice(list(range(SITE_NUM)), p=probs)
            as_set = shuffled_sites[start_site_idx: start_site_idx + num_available_sites[tg_idx] - 1]
            available_sites = {site for site in sites if site.index in as_set}

            tg = TaskGroup(tg_idx, job, num_tasks=num_tasks[tg_idx], available_sites=available_sites)
            tg.real_running_time = task_durations[tg_idx]

            tgs.append(tg)

        job.task_groups = tgs
        job.set_available_sites()

        jobs.append(job)


def print_env():
    print("----------------------- Job Info -----------------------")
    for job in jobs:
        print(job)
    print("\n----------------------- Site Info -----------------------")
    for site in sites:
        print(site)
    print("\n----------------------- Summary -----------------------")
    print("There are {} tasks, {} jobs, {} sites, {} util, {} alpha".format(NUM_TASKS, JOB_NUM, SITE_NUM, UTILIZATION_FACTOR, ALPHA))
    print("MAX_AS_NUM_FOR_TG: {}, MIN_AS_NUM_FOR_TG: {}, MIN_MU: {}, MAX_MU: {}".format(MAX_AS_NUM_FOR_TG, MIN_AS_NUM_FOR_TG, MIN_MU, MAX_MU))


def get_Kg(job: Job):
    """
    Return a map from each site's index to the list of its processable task groups' index.
    """
    return {site.index: [tg.index for tg in job.task_groups if site in tg.available_sites]
            for site in job.available_sites}


def check_validity():
    """
    Check the validity of the generated env.
    """
    for job in jobs:
        assert 0 <= job.index < JOB_NUM
        assert 0 <= job.arrival_time < MAX_ARRIVAL_TU
        assert job.completion_time is None

        for tg in job.task_groups:
            assert 0 <= tg.index < MAX_TG_NUM
            assert tg.job is job
            assert MIN_TASK_NUM_IN_TG <= tg.num_tasks <= MAX_TASK_NUM_IN_TG
            assert tg.num_unfinished_tasks == tg.num_tasks

            if MODE == HOMOGENEOUS_MODE:
                assert tg.real_running_time == 1
            else:
                assert tg.real_running_time >= 1

            assert not (False in [site in job.available_sites for site in tg.available_sites])

        assert not (False in [site in sites for site in job.available_sites])

        Kg = get_Kg(job)
        assert not (False in [0 <= site_idx < SITE_NUM for site_idx in Kg.keys()])
        for site_idx, tg_idx_list in Kg.items():
            for tg_idx in tg_idx_list:
                assert 0 <= tg_idx < len(job.task_groups)
                assert sites[site_idx] in job.task_groups[tg_idx].available_sites

    for site in sites:
        assert 0 <= site.index < SITE_NUM
        assert not (False in [MIN_MU <= cap <= MAX_MU for cap in site.capacities])

    print("Assertion pass!")


def reset():
    """
    Reset the scheduling result to empty.
    """
    for job in jobs:
        job.reset()
        for tg in job.task_groups:
            tg.reset()
    for site in sites:
        site.reset()


# 1. Synthesizing
# create_env()
# check_validity()
# print_env()

# 2. From trace
from_trace()
print_env()
