## TAOS

This repository contains the simulation code of taos (task assignment and order scheduling) 
related algorithms.

We implement 6 algorithms, OBTA, NLIP, WF, RD, OCWF, and OCWF-ACC. The first 4 algorithms 
are First-Come-First-Serve (FIFO) algorithms without adjusting outstanding jobs' orders while the last two algorithms 
are Shortest-Remaining-Time-First (SRTF) algorithms.

### Setup

We use [Alibaba cluster-trace-v2017]([https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md) 
to drive the simulation. We extract a segment from the file `batch_task.scv` in cluster-trace-v2017 
that contains 250 jobs. These jobs include 113653 task instances in total. We derive job arrivals 
and task durations from the timestamps of the recorded task events. We scale the inter-arrival 
times of the jobs to simulate different levels of system utilization from 50% to 75%. 
The default number of sites is 100.

The default settings are with `taos/env.py`. You may change the settings at your wish 
to test the performance and efficiency of the algorithms.

### Run

You can run ``taos/main.py`` directly to obtain the simulation results in default settings. 
The output should be similar to:
```text
----------------------- Summary -----------------------
There are 113653 tasks, 250 jobs, 100 sites, 0.25 util, 0 alpha
MAX_AS_NUM_FOR_TG: 12, MIN_AS_NUM_FOR_TG: 8, MIN_MU: 3, MAX_MU: 5
OBTA progress:: 100%|█████| 250/250 [00:24<00:00, 10.38it/s, jrt=12591]
[OBTA] computation overhead: 24.088769 secs
OBTA: 461.0
NLIP progress:: 100%|█████| 250/250 [00:30<00:00,  8.16it/s, jrt=12599]
[NLIP] computation overhead: 30.652053 secs
NLIP: 483.592
WF progress:: 100%|█████| 250/250 [00:14<00:00, 17.30it/s, jrt=8960]
[WF] computation overhead: 14.448744 secs
WF: 495.94
RD progress:: 100%|█████| 250/250 [00:19<00:00, 13.03it/s, jrt=8960]
[RD] computation overhead: 19.185192 secs
RD: 466.684
OCWF progress:: 100%|█████| 250/250 [12:57<00:00,  3.11s/it, jrt=12627]
[OCWF] computation overhead: 777.408436 secs
OCWF: 328.676
OCWF-ACC progress:: 100%|█████| 250/250 [08:15<00:00,  1.98s/it, jrt=9033]
[OCWF-ACC] computation overhead: 495.733517 secs
OCWF-ACC: 328.676

Average computation overhead:

OBTA: 0.01660825252532959
NLIP: 0.04274543857574463
WF: 0.0003381824493408203
RD: 0.01590808391571045
OCWF: 3.032428377151489
OCWF_ACC: 1.9141362752914428
```
You may use the file ``draw/draw.ipynb`` to obtain the figures of average JRTs, CDF of JRTs, etc.

### Dependencies

See ``requirements.txt``.

The code depends on package `docplex`. You should have a **commercial** or **academic** version 
(NOT the no-cost edition!) of CPLEX optimization studio installed locally (or you have an IBM
Watson Studio Cloud account), and then install the package `docplex` as guided. The programs are 
formulated and solved in `taos/algo/common.py`.
