"""
The scheduler, will be used by FIFO and SJF algorithms.
"""
import env


class ScheduleInfo:
    """
    Anything that related to outstanding jobs or backlogs will be handled by this class.
    """

    def __init__(self):
        # A three-level map that stores the outstanding jobs info for each site
        self.outstanding = {
            # The key is a site, and the value is a map from job to an inner map,
            # where the inner map is from a 3-tuple: (tg, begin_time_slot_index, end_time_slot_index)
            # to num_allocated_tasks. Note that here `time_slot_index` is indexed globally!
            site: {} for site in env.sites
        }

    def reset(self):
        self.outstanding = {
            site: {} for site in env.sites
        }

    def update_estimated_bklg_sizes(self, cur_time):
        """
        Update each site's (estimated) backlog size with the newest schedule info.
        The same to CCGrid paper, we do not use the real backlogs (since in real scenarios
        we can never know the exact completion time of a task in advance), but the estimated
        one in our algorithms, i.e., the number of time units they occupy by assuming
        that each task's running time is one time unit.

        Here the backlog size is the increased num. of time units in terms of current time.
        """
        for site in self.outstanding.keys():
            # Calculated without IJCoE
            estimated_increment_1 = 0
            true_increment = 0
            for tg_with_time2num_allocated_tasks in self.outstanding[site].values():
                # assert len(tg_with_time2num_allocated_tasks) > 0
                for tg_with_time in tg_with_time2num_allocated_tasks.keys():
                    estimated_increment_1 = tg_with_time[1] \
                        if tg_with_time[1] > estimated_increment_1 else estimated_increment_1
                    true_increment = tg_with_time[2] if tg_with_time[2] > true_increment else estimated_increment_1

            # If the first job arrives at some time t > 0, without any backlogged jobs on the site,
            # `estimated_increment_1` will be zero. In this case, `estimated_increment_1 - cur_time` will < 0.
            # To avoid this, we need thw following if-else code
            if estimated_increment_1 < cur_time:
                estimated_increment_1 = 0
            else:
                estimated_increment_1 -= cur_time

            if true_increment < cur_time:
                true_increment = 0
            else:
                true_increment -= cur_time

            # # Calculated with IJCoE
            # estimated_increment_2 = 0
            # for job, tg_with_time2num_allocated_tasks in self.outstanding[site].items():
            #     assert len(tg_with_time2num_allocated_tasks) > 0
            #     num_tasks = sum([num_task for tg_with_time, num_task in tg_with_time2num_allocated_tasks.items()
            #                      if tg_with_time[1] > cur_time])
            #     cap = site.capacities[job.index]
            #     estimated_increment_2 += int(math.ceil(num_tasks / cap))
            # assert estimated_increment_2 >= 0
            #
            # assert estimated_increment_1 >= estimated_increment_2

            # If the task assignment is implemented by task_assignment_with_IJCoE(),
            # we should set `site.estimated_bklg_size` with `estimated_increment_2`
            site.estimated_bklg_size = estimated_increment_1
            site.true_bklg_size = true_increment


info = ScheduleInfo()


def task_assignment(job: env.Job, solution, cur_time):
    """
    Do task assignment for each task of the given job as the solution indicates.
    Currently, intra-job co-execution is not implemented. That is, the tasks of
    different task groups of the same job cannot be put into the same time unit.

    Even though, since all the algorithms are not implemented with IJCoE, the
    comparison is still equitable.
    """
    target_time_units = {site.index: (cur_time + site.estimated_bklg_size + 1) for site in job.available_sites}
    for tg in job.task_groups:
        num_left_tasks = tg.num_unfinished_tasks
        for site in tg.available_sites:
            # This site does not participate in this task group's assignment
            if solution[site.index, tg.index] == 0:
                continue

            if num_left_tasks == 0:
                break

            # For now, we can make sure this site must participate in this job's assignment.
            # Create the value map first
            if job not in info.outstanding[site].keys():
                info.outstanding[site][job] = {}

            # Do the allocation
            num_tasks_in_a_row = site.capacities[job.index]
            while True:
                if num_left_tasks == 0:  # All done
                    break

                if num_left_tasks > num_tasks_in_a_row:
                    # info.outstanding[site][job][(tg, target_time_units[site.index])] = num_tasks_in_a_row
                    info.outstanding[site][job][
                        (tg, target_time_units[site.index], target_time_units[site.index] + tg.real_running_time - 1)
                    ] = num_tasks_in_a_row

                    num_left_tasks -= num_tasks_in_a_row
                else:
                    # info.outstanding[site][job][(tg, target_time_units[site.index])] = num_left_tasks
                    info.outstanding[site][job][
                        (tg, target_time_units[site.index], target_time_units[site.index] + tg.real_running_time - 1)
                    ] = num_left_tasks

                    num_left_tasks = 0

                # Without IJCoE, the target time unit for the next allocation must self-increment
                target_time_units[site.index] += tg.real_running_time


def get_jct(job: env.Job, solution, cur_time, reordered=False):
    """
    Calculate the completion time of the given job.
    """
    if not reordered:
        # The index of time unit (in terms of current time) in which the job completes
        real_increment = 0
        # In FIFO algorithms, current job must be put at the end of the backlogs.
        # Thus, `max(site.estimated_bklg_size)` is exactly the completion time of it
        for site in job.available_sites:
            if sum(solution[site.index, tg.index] for tg in job.task_groups) > 0:
                # real_increment = site.estimated_bklg_size \
                #     if site.estimated_bklg_size > real_increment else real_increment
                real_increment = site.true_bklg_size \
                    if site.true_bklg_size > real_increment else real_increment
        job.completion_time = cur_time + real_increment

    else:
        completion_time = -1
        # In SJF algorithms, current job might not be in the end of the backlogs.
        # Thus, we need to specify its final completion time
        for site in info.outstanding.keys():
            if job in info.outstanding[site].keys():
                # tmp = max(tg_with_time[1] for tg_with_time in info.outstanding[site][job].keys())
                tmp = max(tg_with_time[2] for tg_with_time in info.outstanding[site][job].keys())
                completion_time = tmp if tmp > completion_time else completion_time
        job.completion_time = completion_time


def remove_complete_tasks(job: env.Job, cur_time):
    """
    Remove the complete tasks from `info` and update
    `num_unfinished_tasks` for related task groups.
    """
    for site in job.available_sites:
        if job not in info.outstanding[site].keys():
            continue

        # assert info.outstanding[site][job] != {}

        tmp = {}
        for tg_with_time, num_allocated_tasks in info.outstanding[site][job].items():
            # assert tg_with_time[1] >= cur_time

            # if tg_with_time[1] == cur_time:
            if tg_with_time[2] == cur_time:
                # These tasks complete, just remove them
                tg_with_time[0].num_unfinished_tasks -= num_allocated_tasks
            else:
                tmp[tg_with_time] = num_allocated_tasks

        if tmp == {}:
            # All tasks of this job assigned to this site complete,
            # just remove the job outstanding info from this site
            del info.outstanding[site][job]
        else:
            # Otherwise, update the inner map with the completed tasks removed
            info.outstanding[site][job] = tmp

        # if job in info.outstanding[site].keys():
        #     for tg_with_time in info.outstanding[site][job].keys():
        #         assert tg_with_time[1] > cur_time


def cleanup(jobs, cur_time, processed, pbar):
    """
    Do the cleanup at the end of current time.
    """
    jrts = []
    for job in jobs:
        if job.arrival_time > cur_time:
            continue

        # If a job's arrival time <= t, it must be (re)scheduled before
        # assert job.completion_time is not None

        # Remove the complete tasks
        remove_complete_tasks(job=job, cur_time=cur_time)

        # Update progress bar
        if job.completion_time == cur_time:
            # TODO: Find why assertion failed for sjf algorithms
            # If the job completes at t, according to the actions we have done before,
            # it should have been removed from `info`
            # assert not (False in [job not in info.outstanding[site].keys()
            #                       for site in job.available_sites])

            processed += 1
            jrts.append(job.completion_time - job.arrival_time)

            pbar.set_postfix(jrt='{0}'.format(job.completion_time - job.arrival_time))
            pbar.update(1)

    return jrts
