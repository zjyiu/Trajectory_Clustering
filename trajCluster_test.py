# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
import numpy as np
import time, copy
import matplotlib
matplotlib.use('Agg')
from pyspark import SparkContext,SparkConf
from partition import approximate_trajectory_partitioning, segment_mdl_comp, rdp_trajectory_partitioning, distributed_partition
from point import Point
from cluster import line_segment_clustering, representative_trajectory_generation, line_segment_clustering_original
import json
from matplotlib import pyplot as plt
file_path='hdfs:///PJ/trajCluster.zip'

conf=SparkConf().setAppName('trajCluster').setMaster('spark://master:7077')
#.set('spark.default.parallelism', 50)
sc = SparkContext(conf=conf)
sc.addPyFile(file_path)
#sc=SparkContext()
#print type(sc)
def plot_clustering_result(sc, testcases, norm_cluster, remove_cluster):
    for k, v in remove_cluster.items():
        print("remove cluster: the cluster %d, the segment number %d" % (k, len(v)))

    cluster_s_x, cluster_s_y = [], []
    for k, v in norm_cluster.items():
        cluster_s_x.extend([s.start.x for s in v])
        cluster_s_x.extend([s.end.x for s in v])

        cluster_s_y.extend([s.start.y for s in v])
        cluster_s_y.extend([s.end.y for s in v])
        print("using cluster: the cluster %d, the segment number %d" % (k, len(v)))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    for i in range(len(testcases)):
        source_line_x = [p.x for p in testcases[i]]
        source_line_y = [p.y for p in testcases[i]]
        ax.plot(source_line_x, source_line_y, 'g--', lw=1.0)
        ax.scatter(source_line_x, source_line_y, c='g', alpha=0.5)

    for k, v in norm_cluster.items():
        for s in v:
            _x = [s.start.x, s.end.x]
            _y = [s.start.y, s.end.y]
            if s.traj_id == 1:
                ax.plot(_x, _y, c='k', lw=1.0, alpha=0.7)
            elif s.traj_id == 2:
                ax.plot(_x, _y, c='c', lw=1.0, alpha=0.7)
            elif s.traj_id == 3:
                ax.plot(_x, _y, c='m', lw=1.0, alpha=0.7)
            else:
                ax.plot(_x, _y, c='r', lw=1.0, alpha=0.7)
    ax.scatter(cluster_s_x, cluster_s_y, c='k', alpha=0.5, s=80)

    main_traj_dict = representative_trajectory_generation(sc, norm_cluster, min_lines=2, min_dist=1.0)
    for c, v in main_traj_dict.items():
        v_x = [p.x for p in v]
        v_y = [p.y for p in v]
        ax.plot(v_x, v_y, 'r-', lw=4.0)

    ax.legend()
    plt.savefig("./figures/demo.png", dpi=400)
    plt.show()


def define_testcase(ts):
    ts2 = [s - np.random.randint(10, 20) for s in ts]
    ts3 = [s + np.random.randint(1, 10) for s in ts]
    ts4 = [s + np.random.randint(20, 30) for s in ts]
    traj1 = [Point(ts[i:i+2][0], ts[i:i+2][1]) for i in range(0, len(ts), 2)]
    traj2 = [Point(ts2[i:i + 2][0], ts2[i:i + 2][1]) for i in range(0, len(ts2), 2)]
    traj3 = [Point(ts3[i:i + 2][0], ts3[i:i + 2][1]) for i in range(0, len(ts3), 2)]
    traj4 = [Point(ts4[i:i + 2][0], ts4[i:i + 2][1]) for i in range(0, len(ts4), 2)]
    return [traj1, traj2, traj3, traj4]


def generate_testcases(tss):
    testcases = []
    for ts in tss:
        testcases.extend(define_testcase(ts))
    return testcases


#ts1 = [560.0, 652.0, 543.0, 651.0, 526.0, 649.0, 510.0, 647.0, 494.0, 644.0, 477.0, 639.0, 460.0, 632.0, 446.0, 622.0, 431.0, 611.0, 417.0, 604.0, 400.0, 597.0, 383.0, 587.0, 372.0, 579.0, 363.0, 573.0, 355.0, 563.0, 356.0, 552.0, 361.0, 537.0, 370.0, 523.0, 380.0, 510.0, 391.0, 498.0, 404.0, 485.0, 415.0, 475.0, 429.0, 466.0, 444.0, 459.0, 465.0, 451.0, 493.0, 442.0, 530.0, 432.0, 568.0, 423.0, 606.0, 417.0, 644.0, 412.0, 681.0, 408.0, 714.0, 404.0, 747.0, 401.0, 770.0, 399.0, 793.0, 397.0, 818.0, 395.0 ]
#ts2 = [565.0, 689.0, 547.0, 682.0, 525.0, 674.0, 502.0, 668.0, 480.0, 663.0, 452.0, 660.0, 424.0, 656.0, 400.0, 652.0, 380.0, 650.0, 356.0, 649.0, 335.0, 647.0, 314.0, 642.0, 297.0, 639.0, 283.0, 634.0, 272.0, 625.0, 259.0, 614.0, 245.0, 603.0, 237.0, 596.0, 228.0, 589.0, 218.0, 582.0, 208.0, 574.0, 198.0, 567.0, 193.0, 561.0, 191.0, 554.0, 185.0, 551.0, 181.0, 551.0, 179.0, 549.0, 178.0, 547.0, 178.0, 544.0, 177.0, 540.0, 174.0, 533.0, 170.0, 527.0, 164.0, 523.0, 154.0, 521.0, 145.0, 517.0, 131.0, 514.0, 118.0, 515.0, 106.0, 515.0, 92.0, 512.0, 74.0, 507.0, 57.0, 501.0, 40.0, 495.0, 23.0, 491.0]
#ts3 = [590.0, 495.0, 590.0, 498.0, 593.0, 503.0, 597.0, 507.0, 600.0, 507.0, 602.0, 505.0, 605.0, 497.0, 594.0, 487.0, 580.0, 482.0, 565.0, 483.0, 550.0, 492.0, 547.0, 497.0, 544.0, 499.0, 541.0, 494.0, 540.0, 489.0, 538.0, 479.0, 530.0, 474.0, 528.0, 485.0, 540.0, 480.0, 542.0, 477.0, 543.0, 474.0, 538.0, 476.0, 530.0, 486.0, 524.0, 497.0, 513.0, 507.0, 499.0, 516.0, 482.0, 527.0, 468.0, 538.0, 453.0, 547.0, 438.0, 555.0, 429.0, 563.0, 423.0, 566.0, 420.0, 569.0, 417.0, 572.0, 414.0, 570.0, 411.0, 566.0, 411.0, 557.0, 408.0, 545.0, 405.0, 536.0, 403.0, 530.0, 401.0, 526.0, 401.0, 522.0, 404.0, 523.0, 409.0, 523.0, 418.0, 522.0, 420.0, 522.0, 426.0, 530.0]
#ts4 = [559.0, 492.0, 553.0, 483.0, 548.0, 475.0, 544.0, 466.0, 536.0, 456.0, 534.0, 447.0, 536.0, 438.0, 540.0, 429.0, 551.0, 419.0, 566.0, 408.0, 583.0, 399.0, 600.0, 389.0, 622.0, 379.0, 642.0, 373.0, 660.0, 373.0, 676.0, 375.0, 693.0, 381.0, 708.0, 391.0, 721.0, 402.0, 732.0, 412.0, 737.0, 421.0, 741.0, 429.0, 742.0, 437.0, 738.0, 443.0, 733.0, 447.0, 728.0, 449.0, 722.0, 450.0, 714.0, 451.0, 710.0, 445.0, 700.0, 440.0, 695.0, 440.0, 695.0, 434.0, 700.0, 435.0, 705.0, 436.0, 711.0, 435.0, 708.0, 437.0, 710.0, 440.0, 710.0, 445.0, 705.0, 455.0, 700.0, 462.0, 695.0, 470.0, 690.0, 480.0, 680.0, 490.0, 665.0, 490.0]
#testcases = generate_testcases([ts1, ts2, ts3, ts4])

#file='traj'
#f=open('traj','r')
#testcases=[]
#traj=[]
#while 1:
   #s=f.readline().strip()
   #if(len(s)==0):
   	#break
   #if s=='end':
        #testcases.append(traj)
        #traj=[]
        #continue
   #l=s.split(',')
   #traj.append(Point(int(l[0]),int(l[1])))
#f.close()

file=open('../trajectory.json','r')
t=0
testcases=[]
for line in file.readlines():
        #if(t<=2000):
		#t+=1
                #continue
	if(t==10000):
		break
        data=json.loads(line)
	if(len(data)==0):
		continue
        p_list=[Point(data[i:i + 2][0], data[i:i + 2][1]) for i in range(0, len(data), 2)]
        testcases.append(p_list)
	t=t+1
file.close()

# part 1: partition
#time_start = time.time()
#all_segs1 = []
#for i in range(len(testcases)):
   # all_segs1.extend(approximate_trajectory_partitioning(testcases[i], traj_id=i + 1, theta=6.0))
# part2: clustering
#print("Total segments:", len(all_segs1))
#norm_cluster1, remove_cluster1 = line_segment_clustering_original(all_segs1, min_lines=3, epsilon=15.0)
#time_end = time.time()
#print('Original total time:', time_end-time_start)
#plot_clustering_result(sc, testcases, norm_cluster1, remove_cluster1)
# part 1: partition
#time_start = time.time()
all_segs2 = distributed_partition(sc, testcases)
print("Total segments:", len(all_segs2))
# part2: clustering
norm_cluster2, remove_cluster2 = line_segment_clustering(sc, all_segs2, min_lines=3, epsilon=15.0)
#time_end = time.time()
#print('Distributed total time:', time_end-time_start)
plot_clustering_result(sc, testcases, norm_cluster2, remove_cluster2)



