"""
The OBTA algorithm.
"""
import time
import profiling
from algo import common
from algo import scheduler
from tqdm import tqdm


@profiling.profiling("OBTA")
def OBTA(jobs, sites):
    total = len(jobs)
    processed = 0

    aver_jrt = 0
    JRTs = []
    overheads = []

    with tqdm(total=total) as pbar:
        pbar.set_description("OBTA progress:")
        t = 0
        while True:
            # 1. At the beginning of t, firstly, we check whether all jobs are complete.
            #    If yes, exit the while loop.
            all_complete = True
            for job in jobs:
                if job.arrival_time >= t or job.completion_time is None or job.completion_time >= t:
                    all_complete = False
                    break
            if all_complete:
                break

            # 2. At the beginning of t, if a new job arrives at t, schedule it with current backlog info.
            for job in jobs:
                if job.arrival_time == t:
                    # print("\n\n-----------------------------------------------------------\n" +
                    #       "Start solving for job {0} at t = {1}...".format(job.index, t))

                    solution = {(site.index, tg.index): 0 for tg in job.task_groups for site in job.available_sites}

                    s = time.time()
                    # Call obta() to obtain the task assignment solution and do it
                    _ = common.obta(job=job, solution=solution)
                    e = time.time()
                    overheads.append(e - s)

                    # Do real assignment with the solution obtained for chosen job.
                    # This will really update the outstanding job info of the involved sites
                    scheduler.task_assignment(job=job, solution=solution, cur_time=t)
                    scheduler.info.update_estimated_bklg_sizes(cur_time=t)

                    # Set its completion time
                    scheduler.get_jct(job=job, solution=solution, cur_time=t)

            # Running the tasks... Now we are at the end of time t.

            # 3. Do the update at the end of t. Specifically, for each job arrived <= t, do:
            #    (1) Remove the tasks that complete in t;
            #    (2) Mark a job as complete if its completion time is exactly t.
            jrts = scheduler.cleanup(jobs=jobs, cur_time=t, processed=processed, pbar=pbar)
            aver_jrt += sum(jrts)
            JRTs.extend(jrts)

            t += 1

            # Do not forget to update `site.estimated_bklg_size` since we are moving from t to t+1
            # TODO: There are two ways to update it. Check the validity
            # info.update_estimated_bklg_sizes(cur_time=cur_time + 1)
            for site in sites:
                if site.estimated_bklg_size > 0:
                    site.estimated_bklg_size -= 1

    aver_jrt /= len(jobs)
    # print("\n\nAverage JRT: {0}\n-----------------------------------------------------------".format(aver_jrt))
    return aver_jrt, JRTs, overheads
