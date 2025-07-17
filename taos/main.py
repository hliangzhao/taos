"""
The simulation entry.
"""
import time
import env
from algo.fifo.obta import OBTA
from algo.fifo.nlip import NLIP
from algo.fifo.lip import LIP
from algo.fifo.wf import WF
from algo.fifo.rd import RD
from algo.sjf.ocwf import OCWF
from algo.sjf.ocwf_acc import OCWF_ACC
from algo.fifo.obta_1 import OBTA_1
from algo.fifo.obta_2 import OBTA_2
from algo import scheduler

# 这个函数是文件保存函数，第一个参数应当是系统利用率，第二个是奇普夫参数，接下来是四个参数：两个
# 这里存了六组数字，分别对应六个算法，记录了jobs的响应时间，其实对应的是性能
def save_JRTs(obta_jrts, nlip_jrts, wf_jrts, rd_jrts, ocwf_jrts, ocwf_acc_jrts):
    filename = "JRTs_u{}_zipf{}_minAS{}_maxAS{}_minmu{}_maxmu{}".format(env.UTILIZATION_FACTOR, env.ALPHA, env.MIN_AS_NUM_FOR_TG, env.MAX_AS_NUM_FOR_TG, env.MIN_MU, env.MAX_MU)
    with open(filename, "w") as f:
        f.writelines(str(i) + " " for i in obta_jrts)
        f.writelines("\n")
        f.writelines(str(i) + " " for i in nlip_jrts)
        f.writelines("\n")
        f.writelines(str(i) + " " for i in wf_jrts)
        f.writelines("\n")
        f.writelines(str(i) + " " for i in rd_jrts)
        f.writelines("\n")
        f.writelines(str(i) + " " for i in ocwf_jrts)
        f.writelines("\n")
        f.writelines(str(i) + " " for i in ocwf_acc_jrts)
        f.writelines("\n")
# 这个函数也是文件保存函数，第一个参数应当是系统利用率，第二个是奇普夫参数，接下来是四个参数：两个
# 这里存了六组数字，分别对应六个算法的平均响应时间和平均开销，其实对应的是算法开销
def save_aver_JRT_and_overhead(obta_ret, obta_overheads, nlip_ret, nlip_overheads,
                               wf_ret, wf_overheads, rd_ret, rd_overheads, ocwf_ret, ocwf_overheads,
                               ocwf_acc_ret, ocwf_acc_overheads):
    filename = "res_u{}_zipf{}_minAS{}_maxAS{}_minmu{}_maxmu{}".format(env.UTILIZATION_FACTOR, env.ALPHA, env.MIN_AS_NUM_FOR_TG, env.MAX_AS_NUM_FOR_TG, env.MIN_MU, env.MAX_MU)
    with open(filename, "w") as f:
        f.writelines("{} {}\n".format(obta_ret, sum(obta_overheads) / env.JOB_NUM))
        f.writelines("{} {}\n".format(nlip_ret, sum(nlip_overheads) / env.JOB_NUM))
        f.writelines("{} {}\n".format(wf_ret, sum(wf_overheads) / env.JOB_NUM))
        f.writelines("{} {}\n".format(rd_ret, sum(rd_overheads) / env.JOB_NUM))
        f.writelines("{} {}\n".format(ocwf_ret, sum(ocwf_overheads) / env.JOB_NUM))
        f.writelines("{} {}\n".format(ocwf_acc_ret, sum(ocwf_acc_overheads) / env.JOB_NUM))


# ----------------------------------------------------------------------------
# Takeaways:
# 1. Generally, WF is pretty good.
#
# 2. OBTA and NLIP perform similarly, but OBTA is much faster.
#    That is, our policy works.
#
# 3. Without inter-job co-execution, the "waste" of computing slots
#    in a time unit can be large if the computing capacities are large.
#    It leads to the result that WF performs very similar to OBTA, even smaller sometimes.
# ----------------------------------------------------------------------------
# 这里实际上只执行了一次，env.reset()和scheduler.info.reset()都是重置环境
for i in range(0, 1):
    env.reset()
    scheduler.info.reset()

    obta_ret, obta_jrts, obta_overheads = OBTA(env.jobs, env.sites)
    # 这里env.jobs是工作集合，env.sites是服务器集合
    # OBTA算法调用，这里jrts是各个作业响应时间，overheads是计算开销，这两个变量肯定都是多值的，每个值对应于某个job
    print("OBTA: {}".format(obta_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    nlip_ret, nlip_jrts, nlip_overheads = NLIP(env.jobs, env.sites)
    print("NLIP: {}".format(nlip_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    lip_ret, lip_jrts, lip_overheads = LIP(env.jobs, env.sites)
    print("LIP: {}".format(lip_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    obta_1_ret, obta_1_jrts, obta_1_overheads = OBTA_1(env.jobs, env.sites)
    print("OBTA_1: {}".format(obta_1_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    obta_2_ret, obta_2_jrts, obta_2_overheads = OBTA_2(env.jobs, env.sites)
    print("OBTA_2: {}".format(obta_2_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    wf_ret, wf_jrts, wf_overheads = WF(env.jobs, env.sites)
    print("WF: {}".format(wf_ret))
    env.reset()
    scheduler.info.reset()

    time.sleep(1)
    rd_ret, rd_jrts, rd_overheads = RD(env.jobs, env.sites)
    print("RD: {}".format(rd_ret))
    env.reset()
    scheduler.info.reset()

    # time.sleep(1)
    # ocwf_ret, ocwf_jrts, ocwf_overheads = OCWF(env.jobs, env.sites)
    # print("OCWF: {}".format(ocwf_ret))
    # env.reset()
    # scheduler.info.reset()

    # time.sleep(1)
    # ocwf_acc_ret, ocwf_acc_jrts, ocwf_acc_overheads = OCWF_ACC(env.jobs, env.sites)
    # print("OCWF-ACC: {}".format(ocwf_acc_ret))
    # env.reset()
    # scheduler.info.reset()

    # 这里是根据上面得到的开销计算平均开销
    # print("\nAverage computation overhead:\n")
    # print("OBTA: {0}\nNLIP: {1}\nWF: {2}\nRD: {3}\nOCWF: {4}\nOCWF_ACC: {5}\nLIP:{6}\nLIP:{7}\nOBTA_1:{8}\nOBTA_2:{9}".format(
    #     sum(obta_overheads) / env.JOB_NUM,
    #     sum(nlip_overheads) / env.JOB_NUM,
    #     sum(wf_overheads) / env.JOB_NUM,
    #     sum(rd_overheads) / env.JOB_NUM,
    #     sum(ocwf_overheads) / env.JOB_NUM,
    #     sum(ocwf_acc_overheads) / env.JOB_NUM,
    #     sum(lip_overheads) / env.JOB_NUM,
    #     sum(obta_1_overheads) / env.JOB_NUM,
    #     sum(obta_2_overheads) / env.JOB_NUM
    #     ))

    # print("\nAverage computation overhead:\n")
    # print("OBTA: {0}\nNLIP: {1}\nWF: {2}\nRD: {3}\nOCWF: {4}\nOCWF_ACC: {5}\nLIP:{6}".format(
    #     sum(obta_overheads) / env.JOB_NUM,
    #     sum(nlip_overheads) / env.JOB_NUM,
    #     sum(wf_overheads) / env.JOB_NUM,
    #     sum(rd_overheads) / env.JOB_NUM,
    #     sum(ocwf_overheads) / env.JOB_NUM,
    #     sum(ocwf_acc_overheads) / env.JOB_NUM,
    #     sum(lip_overheads) / env.JOB_NUM))
#这里是对结果进行保存
# save_JRTs(obta_jrts, nlip_jrts, wf_jrts, rd_jrts, ocwf_jrts, ocwf_acc_jrts)

# save_aver_JRT_and_overhead(obta_ret, obta_overheads, nlip_ret, nlip_overheads,
#                            wf_ret, wf_overheads, rd_ret, rd_overheads, ocwf_ret, ocwf_overheads,
#                            ocwf_acc_ret, ocwf_acc_overheads)

