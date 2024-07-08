"""
The common function modules.
"""
import math
from queue import PriorityQueue
from docplex.mp.model import Model

import env


def competence(c: int, bklg: int):
    return max(c - bklg, 0)


# ----------------------------------------------------------------------------
# Programming related functions.
# ----------------------------------------------------------------------------

def create_coeff_matrix(job: env.Job):
    """
    Return the coefficient matrix for the given job g (of size S_g x K_g).

    Define A = [[0, 1, ...], [1, 1, ...], [1, 0, ...]] of size S_g x K_g.
    For each row m \in S_g, k \in K_g: A[m, k] = 1 if k \in K_g^m, else 0.
    """
    Kg = env.get_Kg(job)

    A = {}  # of size S_g x K_g
    for site in job.available_sites:
        m = site.index
        for tg in job.task_groups:
            k = tg.index
            A[(m, k)] = 1 if k in Kg[m] else 0

    return A


def C_min(job: env.Job):
    ret = -1
    for tg in job.task_groups:
        bklgs = [site.estimated_bklg_size for site in tg.available_sites]
        caps = [site.capacities[tg.job.index] for site in tg.available_sites]

        xi_k = x_k(num_tasks_to_allocate=tg.num_unfinished_tasks, bklgs=bklgs, caps=caps)
        ret = xi_k if xi_k > ret else ret

    return ret


def C_max(job: env.Job):
    ret = -1
    for site in job.available_sites:
        num_total_proc_tasks = sum([tg.num_unfinished_tasks for tg in job.task_groups if site in tg.available_sites])
        tmp = math.ceil(num_total_proc_tasks / site.capacities[job.index]) + site.estimated_bklg_size
        ret = tmp if tmp > ret else ret

    return ret


def subranges(job: env.Job):
    """
    Divide the search range [C_min, C_max] into a list of subranges and return it.
    """
    bklgs = sorted([site.estimated_bklg_size for site in job.available_sites])
    c_min = C_min(job)
    c_max = C_max(job)

    # `smallest` is the smallest index on which the element is larger than c_min
    # `largest` is the smallest index on which the element is larger than c_max
    smallest = -1
    largest = -1
    for (idx, bklg) in enumerate(bklgs):
        if c_min < bklg:
            smallest = idx
            break

    if smallest == -1:
        # c_min >= all bklgs, no division is required
        return [[c_min, c_max]]

    for (idx, bklg) in enumerate(bklgs):
        if idx < smallest:
            continue
        if c_max < bklg:
            largest = idx
            break

    if smallest == largest:
        # In this case, there is an index n,
        # such that bklg_n <= c_min <= c_max < bklg_{n+1} = smallest = largest.
        return [[c_min, c_max]]

    if largest == -1:
        largest = len(bklgs)
    # Return [[c_min, bklg_m], [bklg_m, bklg_{m+1}], ..., [bklg_{n-1}, bklg_n], [bklg_n, c_max]],
    # where bklg_m's index is `smallest`, and bklg_n's index is `largest` - 1
    ret = [[c_min, bklgs[smallest]]]
    ret.extend([[bklgs[i], bklgs[i + 1]] for (i, _) in enumerate(bklgs) if smallest <= i <= largest - 2])
    ret.append([bklgs[largest - 1], c_max])

    # TODO [Optimize]: There could be repeated single-value range such as [2, 4], [4, 4], [4, 4], ...
    #  Find a way to remove the redundant ones.

    return ret


def obta(job: env.Job, solution):
    """
    The OBTA algorithm to obtain the assignment solution for the given job.
    """
    estimated = -1

    ranges = subranges(job)
    solved = False

    # Get the coefficient matrix
    A = create_coeff_matrix(job)

    for r in ranges:
        # ----------------------------------------------------------------------------
        # Solve the ILP for each subrange:
        #
        #               min_{ f_{km}, c } c
        # s.t. \sum_{k \in K_g^m} f_{km} - c \leq -b_m,         \forall m \in Set of used sites
        #      \sum_{k \in K_g^m} f_{km} \leq 0,                \forall m \in Set of unused sites
        #      |T_g^k| \leq \sum_{m \in S_g^k} \mu_m f_{km},    \forall k \in [K_g]
        #                  0 \leq f_{km},                       \forall m \in S_g, k \in [K_g^m]
        #              r[0] \leq c \leq r[1]
        # ----------------------------------------------------------------------------

        # Create integer linear programming (ILP) model
        mdl = Model(name="ILP-subrange-{0}-{1}".format(r[0], r[1]))

        # Create decision variables (S_g x K_g + 1 vars)
        flows = {(site.index, tg.index): mdl.integer_var(lb=0, name="flow-m{0}-k{1}".format(site.index, tg.index))
                 for tg in job.task_groups for site in job.available_sites}
        c = mdl.integer_var(lb=r[0], ub=r[1], name="C")

        # Create constraints (S_g + K_g cons)
        # for site in job.available_sites:
        #     # No backlog can be in (r[0], r[1])
        #     assert (site.estimated_bklg_size <= r[0] or site.estimated_bklg_size >= r[1])

        # For used sites
        _ = {site.index: mdl.add_constraint(
            ct=mdl.sum(A[site.index, tg.index] * flows[site.index, tg.index]
                       for tg in job.task_groups) - c <= -site.estimated_bklg_size,
            ctname="cons-cpt-used-site{0}".format(site.index)
        ) for site in job.available_sites if site.estimated_bklg_size <= r[0]}

        # For unused sites
        _ = {site.index: mdl.add_constraint(
            ct=mdl.sum(A[site.index, tg.index] * flows[site.index, tg.index] for tg in job.task_groups) <= 0,
            ctname="cons-cpt-used-site{0}".format(site.index)
        ) for site in job.available_sites if site.estimated_bklg_size >= r[1]}

        # For task groups
        _ = {tg.index: mdl.add_constraint(
            ct=mdl.sum(site.capacities[job.index] * flows[site.index, tg.index]
                       for site in tg.available_sites) >= tg.num_tasks,
            ctname="cons-proc-tsk-tg{0}".format(tg.index)
        ) for tg in job.task_groups}

        # Minimization
        mdl.minimize(c)
        # mdl.print_information()

        if mdl.solve():
            # print("* Solved!")
            solved = True
            # mdl.print_solution()

            # Save the solution
            mdl_result2solution(job, flows, solution)
            estimated = c.solution_value
            break
        # else:
        # print("* No solution! Test next subrange...\n")

    assert solved
    return estimated


def nlip(job: env.Job, solution):
    """
    Solve the NLIP directly:

                   min_{ f_{km}, c } c
    s.t. \sum_{k \in K_g^m} f_{km} \leq max \{ c - b_m, 0 \},    \forall m \in S_g
        |T_g^k| \leq \sum_{m \in S_g^k} \mu_m f_{km},            \forall k \in [K_g]
                      0 \leq f_{km},                             \forall m \in S_g, k \in [K_g^m]
                      c \geq 0
    """
    estimated = -1

    solved = False

    # Get the coefficient matrix
    A = create_coeff_matrix(job)

    # Create ILP model
    mdl = Model(name="NLIP")
    # mdl.print_information()

    # Create decision variables (S_g x K_g + 1 vars)
    flows = {(site.index, tg.index): mdl.integer_var(lb=0, name="flow-m{0}-k{1}".format(site.index, tg.index))
             for tg in job.task_groups for site in job.available_sites}
    c = mdl.integer_var(lb=0, name="C")

    # Create constraints (S_g + K_g cons)
    for site in job.available_sites:
        mdl.add_constraint(
            ct=mdl.sum(A[site.index, tg.index] * flows[site.index, tg.index]
                       # NOTE: Here is the difference!
                       for tg in job.task_groups) <= mdl.max(c - site.estimated_bklg_size, 0),
            ctname="cons-site{0}".format(site.index)
        )
    for tg in job.task_groups:
        mdl.add_constraint(
            ct=mdl.sum(site.capacities[job.index] * flows[site.index, tg.index]
                       for site in tg.available_sites) >= tg.num_tasks,
            ctname="cons-tg{0}".format(tg.index)
        )

    # Minimization
    mdl.minimize(c)
    # mdl.print_information()

    if mdl.solve():
        # print("* Solved!")
        solved = True
        # mdl.print_solution()

        # Save the solution.
        mdl_result2solution(job, flows, solution)
        estimated = c.solution_value
    else:
        print("* No solution!\n")

    assert solved
    return estimated


def mdl_result2solution(job: env.Job, flows, solution):
    """
    Decoding the result obtained by programming into `solution`.
    """
    # # Check feasibility
    # for site in job.available_sites:
    #     for tg in job.task_groups:
    #         if site not in tg.available_sites:
    #             assert int(flows[site.index, tg.index].solution_value) == 0

    # Save the result
    for site in job.available_sites:
        for tg in job.task_groups:
            solution[site.index, tg.index] = int(flows[site.index, tg.index].solution_value)


# ----------------------------------------------------------------------------
# Water-filling related functions.
# ----------------------------------------------------------------------------

def x_k(num_tasks_to_allocate, bklgs, caps):
    """
    The calculation of x_k for any task group k of some job.
    """
    x_k = min(bklgs) + 1
    while True:
        if sum(competence(x_k, bklg) * cap for (bklg, cap) in zip(bklgs, caps)) >= num_tasks_to_allocate:
            break
        else:
            x_k += 1

    return x_k


def bs_x_k(num_tasks_to_allocate, bklgs, caps):
    """
    The calculation of x_k through binary search for any task group k of some job.
    """
    lower_bound = min(bklgs) + 1
    upper_bound = max(bklgs) + num_tasks_to_allocate
    x_k = lower_bound
    while lower_bound < upper_bound:
        x_k = int((lower_bound + upper_bound) / 2) #round down
        if sum(competence(x_k, bklg) * cap for (bklg, cap) in zip(bklgs, caps)) >= num_tasks_to_allocate:
            #valid
            upper_bound = x_k
        else:
            lower_bound = x_k + 1

    return lower_bound


def wf(job: env.Job, tmp_bklgs, solution):
    """
    The water-filling heuristic to obtain the assignment solution for the given job.
    """
    estimated = -1

    for tg in job.task_groups:
        bklgs = [tmp_bklgs[site.index] for site in tg.available_sites]
        caps = [site.capacities[job.index] for site in tg.available_sites]

        #xi_k = x_k(num_tasks_to_allocate=tg.num_unfinished_tasks, bklgs=bklgs, caps=caps)
        xi_k = bs_x_k(num_tasks_to_allocate=tg.num_unfinished_tasks, bklgs=bklgs, caps=caps)

        for site in tg.available_sites:
            increment = xi_k - tmp_bklgs[site.index]
            if increment > 0:
                solution[site.index, tg.index] = increment
                tmp_bklgs[site.index] = xi_k
                estimated = xi_k if xi_k > estimated else estimated
            else:
                solution[site.index, tg.index] = 0

    return estimated

def rd(job: env.Job, tmp_bklgs, solution):
    """
    The replica-deletion heuristic to obtain the assignment solution for the given job.
    step1: Add tasks to available sites, and put sites into priority queue QA with backlog size as priority.
    step2: For each site, we maintain the number of replicas of tasks using a priority queue QB_i.
    step3: Meanwhile, we maintain a global table that records the remaining replicas of each task.
    step4: Pop items from priority queue QA to delete tasks, and lazy update items of QB_i using the global table.

    Complexity: O(Mlog(M)+n*M*(log(M)+M*log(n)) = O(M^2 * nlog(n))
    """

    #format: (-redundant_backlog_size, -initial_backlog_size, site_index, priority queue)
    sites_info = PriorityQueue()
    for site_idx, bklgs_size in tmp_bklgs.items():
        item = [-bklgs_size, -bklgs_size, site_idx, PriorityQueue()]
        for tg in job.task_groups:
            for site in tg.available_sites:
                if site.index == site_idx:
                    item[0] -= tg.num_unfinished_tasks
                    #format: (-tg_num_replicas_on_all_sites, -tg_num_replicas_on_this_site, tg_idx)
                    item[3].put([-(tg.num_unfinished_tasks * len(tg.available_sites)), -tg.num_unfinished_tasks, tg.index])
        sites_info.put(item)

    #the total number of replicas of task groups
    tg_num_replicas = {}
    tg_num_tasks = {}
    for tg in job.task_groups:
        tg_num_replicas[tg.index] = tg.num_unfinished_tasks * len(tg.available_sites)
        tg_num_tasks[tg.index] = tg.num_unfinished_tasks

    job_caps = {}
    for site in job.available_sites:
        job_caps[site.index] = site.capacities[job.index]

    while (sites_info.empty() == False):
        si = sites_info.get()
        site_idx = si[2]
        cap = job_caps[site_idx]
        #try to delete a task
        while (si[3].empty() == False):
            item = si[3].get()
            tg_idx = item[2]
            if -item[0] != tg_num_replicas[tg_idx]:
                #lazy update, re-enter the priority queue
                item[0] = -tg_num_replicas[tg_idx]
                si[3].put(item)
            else:
                #item has the latest information
                #delete_num = 1
                delete_num = max(1, min(cap, -item[1], tg_num_replicas[tg_idx] - tg_num_tasks[tg_idx]))
                if tg_num_replicas[tg_idx] - delete_num >= tg_num_tasks[tg_idx]:
                    #delete a task of this tg
                    si[0] += delete_num
                    tg_num_replicas[tg_idx] -= delete_num
                    item[0] += delete_num
                    item[1] += delete_num
                    if item[1] < 0:
                        #there are still replicas on this site
                        si[3].put(item)
                    break
                else:
                    #otherwise: can not delete any more, pop from the priority queue
                    solution[site_idx, tg_idx] = math.ceil(-item[1] / cap)

        if (si[3].empty() == False):
            sites_info.put(si)

    #for tg in job.task_groups:
     #   print("rd tg: ", tg)
    #print("rd sol: ", solution)

    return 0