"""
Implementation of Order-Conscious Balanced Task Assignment (OCBTA) with WF as the backend
but accelerated. Here a full order of all the outstanding jobs is calculated.
"""
import time
import math
import profiling
from algo import common
from algo import scheduler
from tqdm import tqdm


@profiling.profiling("OCWF-ACC")
def OCWF_ACC(jobs, sites):
    """
    Implementation of OCWF-ACC, and a full order of all the outstanding jobs
    is calculated when a new job arrives.
    """
    total = len(jobs)
    processed = 0

    aver_jrt = 0
    JRTs = []
    overheads = []

    with tqdm(total=total) as pbar:
        pbar.set_description("OCWF-ACC progress:")
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

            for job in jobs:
                # 2. At the beginning of t, when a new job arrives at t, order scheduling is triggered.
                #    Retrieve all the outstanding jobs and schedule them with the earliest completion time first.
                #    For each job, task assignment and completion time is computed by WF.
                if job.arrival_time == t:
                    # Ot is a list of outstanding jobs to a boolean indicating it is "reordered & rescheduled" or not
                    Ot = {oj: False for oj in jobs
                          # If oj is exactly the job arrives at t, its `completion_time` is None.
                          # Any earlier arrived job must have a non-None `completion_time`
                          # TODO: Check the difference
                          # if (oj.arrival_time <= t and (oj.completion_time is None or oj.completion_time >= t))}
                          if oj.arrival_time <= t and (sum(tg.num_unfinished_tasks) > 0 for tg in oj.task_groups)}

                    # Qt is the sorted version of Ot by their estimated completion time
                    Qt = []

                    # We need to re-schedule all the incomplete jobs.
                    # Thus, the existing allocations are discarded, including the outstanding info and backlog sizes
                    global_bklgs = {site.index: 0 for site in sites}
                    scheduler.info.reset()

                    s = time.time()
                    while len(Qt) < len(Ot):
                        # During each while loop, we select the next job and schedule it
                        min_estimated = math.inf
                        chosen_job = None

                        # We need to store the backlog changes made by each job's assignment.
                        # If a job is chosen, its stored info will be used to update the global outstanding info
                        virtual_bklgs = {
                            job: {site.index: global_bklgs[site.index] for site in job.available_sites}
                            for job in Ot if job not in Qt
                        }

                        # We also need to store the solutions made by each job's assignment.
                        # If a job is chosen, its stored solution will be adopted as its true assignment solution
                        virtual_solutions = {
                            job: {(site.index, tg.index): 0 for tg in job.task_groups for site in job.available_sites}
                            for job in Ot if job not in Qt
                        }

                        # Select the next earliest job to append to Qt
                        for oj, rescheduled in Ot.items():
                            if rescheduled:
                                continue

                            # ---------------------------------------------------------------
                            # Dive into the calculation of WF
                            estimated = -1
                            larger_than_min = False

                            for tg in oj.task_groups:
                                bklgs = [virtual_bklgs[oj][site.index] for site in tg.available_sites]
                                caps = [site.capacities[job.index] for site in tg.available_sites]

                                xi_k = common.x_k(num_tasks_to_allocate=tg.num_unfinished_tasks, bklgs=bklgs, caps=caps)

                                # If c_min (max \{ xi_k \} of current oj is larger than `min_estimated`,
                                # it cannot be the next reordered job, just skip it
                                if xi_k > min_estimated:
                                    larger_than_min = True
                                    break

                                for site in tg.available_sites:
                                    increment = xi_k - virtual_bklgs[oj][site.index]
                                    if increment > 0:
                                        virtual_solutions[oj][site.index, tg.index] = increment
                                        virtual_bklgs[oj][site.index] = xi_k
                                        estimated = xi_k if xi_k > estimated else estimated

                                    else:
                                        virtual_solutions[oj][site.index, tg.index] = 0

                            if larger_than_min:
                                continue
                            # ---------------------------------------------------------------

                            if estimated < min_estimated:
                                min_estimated = estimated
                                chosen_job = oj

                        Qt.append(chosen_job)
                        Ot[chosen_job] = True

                        # Do real assignment with the solution obtained for chosen job
                        # This will really update the outstanding job info of the involved sites
                        scheduler.task_assignment(job=chosen_job,
                                                  solution=virtual_solutions[chosen_job],
                                                  cur_time=t)
                        # Update each site's backlog with sched.info
                        scheduler.info.update_estimated_bklg_sizes(cur_time=t)

                        # Set its completion time
                        scheduler.get_jct(job=job, solution=virtual_solutions[chosen_job], cur_time=t, reordered=True)

                        # Update the real outstanding for the involved sites
                        for site in chosen_job.available_sites:
                            global_bklgs[site.index] = site.estimated_bklg_size

                    # Every oj in Ot should have been rescheduled
                    # assert not (False in [rescheduled for rescheduled in Ot.values()])
                    e = time.time()
                    overheads.append(e - s)

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
