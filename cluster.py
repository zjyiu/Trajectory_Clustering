import math
from pyspark import SparkContext
from segment import compare, Segment, show_seg
from point import Point
from collections import deque, defaultdict

min_traj_cluster = 2  # 定义聚类的簇中至少需要的trajectory数量


def cal_neighbor(seg, seg_tmp, epsilon):
    """计算两个segment距离是否在epsilon范围内，是则是邻居，否则不是
    parameter
    ---------
        seg: 第一个segment
        seg_tmp: 第二个segment
        epsilon: float, segment之间的距离度量阈值
    return
    ------
        True or False, 返回两个segment是否是邻居.
    """
    seg_long, seg_short = compare(seg, seg_tmp) # 长线段在前，短线段在后
    if seg_long.get_all_distance(seg_short) <= epsilon:
        return True
    else:
        return False

def neighborhood(sc, seg, segs, epsilon=2.0):
    """分布式计算一个segment在距离epsilon范围内的所有segment集合, 计算的时间复杂度为O(n). n为所有segment的数量。
    将segment集合分成多片，分布式地计算它们与原segment是否是邻居关系，最后得到所有是邻居关系的segmentID，reduce为最终的邻居segment集合
    parameter
    ---------
        sc: SparkContext()
        seg: Segment instance, 原segment
        segs: List[Segment, ...], 所有的segment集合, 为所有集合的partition分段结果集合
        epsilon: float, segment之间的距离度量阈值
    return
    ------
        List[segment, ...], 返回seg在距离epsilon内的所有Segment集合.
    """
    rdd_segment = sc.parallelize([(segs[i], i) for i in range(len(segs))])
    rdd_neighbor = rdd_segment.filter(lambda seg_temp: cal_neighbor(seg, seg_temp[0], epsilon)) #分布式选出是邻居的segment
    rdd_neighbor_id = rdd_neighbor.map(lambda seg_index: [seg_index[1]]) #返回这些semgent的ID
    result = rdd_neighbor_id.reduce(lambda p1,p2:p1+p2)
    return result

def expand_cluster(sc, segs, queue, cluster_id, epsilon, min_lines):
    """线段segment聚类中的簇扩大过程，设置一个队列，循环地对队列中的所有segment检查邻居，并且把邻居也放进队列中，直到队列中无segment。
    parameter
    ---------
        sc: SparkContext()
        segs: List[Segment, ...], 所有轨迹的partition划分后的segment集合.
        queue: 需要扩大的原簇
        epsilon: float, segment之间的距离度量阈值
        min_lines: int or float, 轨迹在epsilon范围内的segment数量的最小阈值
    """
    while len(queue) != 0:
        curr_seg = queue.popleft()
        curr_num_neighborhood = neighborhood(sc, curr_seg, segs, epsilon)
        if len(curr_num_neighborhood) >= min_lines:
            for m in curr_num_neighborhood:
                if segs[m].cluster_id == -1:
                    queue.append(segs[m])
                    segs[m].cluster_id = cluster_id
        else:
            pass

def line_segment_clustering(sc, traj_segments, epsilon = 2.0, min_lines= 5):
    """线段segment聚类, 采用dbscan的聚类算法, 参考论文中的伪代码来实现聚类, 论文中的part4.2部分中的伪代码及相关定义。
    论文中的线段聚类算法分为两层循环：第一层对于每一条线段计算邻居并进行簇的扩大过程，在此过程中所有的线段都会被归类进该簇，后续不需要计算。
                                    第二层是邻居计算过程(簇扩大过程本质上也是邻居的计算)
    两层都需要完整遍历整个segment集合，因此其时间复杂度为O(N^2)

    实现聚类算法的分布式，我们采用对第二层遍历进行分布式处理，而第一层串行处理，之所以采用这种方法是因为：
    1、外层循环在执行过程中会对于整个segment集合进行修改，分片后的数据在不同片之间存在依赖，因此无法进行分布式处理。
    2、内层循环是简单遍历整个segment集合，分别计算一次距离，易于分布式的实现，而外层循环的每一次执行中，内层循环的执行占到了绝大多数的工作量，对内层循环进行分布式处理也能有效提高处理效率

    该聚类算法的时间复杂度为O(N^2)，且算法的两层循环都需要遍历全集合，即使进行分布式处理后执行时间仍然会随着N的增长快速膨胀。
    parameter
    ---------
        sc: SparkContext()
        traj_segments: List[Segment, ...], 所有轨迹的partition划分后的segment集合.
        epsilon: float, segment之间的距离度量阈值
        min_lines: int or float, 轨迹在epsilon范围内的segment数量的最小阈值
    return
    ------
        Tuple[Dict[int, List[Segment, ...]], ...], 返回聚类的集合和不属于聚类的集合, 通过dict表示, key为cluster_id, value为segment集合
    """
    cluster_id = 0
    cluster_dict = defaultdict(list)
    for seg in traj_segments:
        _queue = deque(list(), maxlen=50)
        if seg.cluster_id == -1:
            seg_num_neighbor_set = neighborhood(sc, seg, traj_segments, epsilon=epsilon)  # 该步是分布式的
            if len(seg_num_neighbor_set) >= min_lines:
                seg.cluster_id = cluster_id
                for sub_seg_index in seg_num_neighbor_set:
                    sub_seg = traj_segments[sub_seg_index]
                    sub_seg.cluster_id = cluster_id  # 邻居全部属于该簇
                    _queue.append(sub_seg) # 需要扩大的簇
                expand_cluster(sc, traj_segments, _queue, cluster_id, epsilon, min_lines)
                cluster_id += 1
            else:
                seg.cluster_id = -1
        if seg.cluster_id != -1:
            cluster_dict[seg.cluster_id].append(seg)  # 将轨迹放入到聚类的集合中, 按dict进行存放

    remove_cluster = dict()
    cluster_number = len(cluster_dict)
    for i in range(0, cluster_number):
        traj_num = len(set(map(lambda s: s.traj_id, cluster_dict[i])))  # 计算每个簇下的轨迹数量
        print("the %d cluster lines:" % i, traj_num)
        if traj_num < min_traj_cluster:
            remove_cluster[i] = cluster_dict.pop(i)
    return cluster_dict, remove_cluster


def segment_transformation(seg, sin_theta, cos_theta):
    s, e = seg.start, seg.end
    return Segment(
                Point(s.x * cos_theta + s.y * sin_theta, s.y * cos_theta - s.x * sin_theta, -1),
                Point(e.x * cos_theta + e.y * sin_theta, e.y * cos_theta - e.x * sin_theta, -1),
                traj_id=seg.traj_id,
                cluster_id=seg.cluster_id)


def calc_start_y(seg, p, sort_point):
    s, e = seg.start, seg.end
    if (sort_point[p].x <= e.x) and (sort_point[p].x >= s.x):
        if s.x == e.x:
            return Point(0, 0, -1)
        elif s.y == e.y:
            return Point(sort_point[p].x, s.y, -1)
        else:
            return Point(sort_point[p].x, (e.y - s.y) / (e.x - s.x) * (sort_point[p].x - s.x) + s.y, -1)
    return Point(0, 0, -1)


def calc_intersect_cnt(seg, p, sort_point):
    s, e = seg.start, seg.end
    if (sort_point[p].x <= e.x) and (sort_point[p].x >= s.x):
        if s.x == e.x:
            return 0
        else:
            return 1
    return 0

def representative_trajectory_generation(sc, cluster_segment, min_lines = 3, min_dist = 2.0):
    """通过论文中的算法对轨迹进行变换, 提取代表性路径, 在实际应用中必须和当地的路网结合起来, 提取代表性路径, 该方法就是通过算法生成代表性轨迹
    由于该变换和提取过程中的计算非常的基础，很容易分布式化。
    parameter
    ---------
        sc: SparkContext()
        cluster_segment: Dict[int, List[Segment, ...], ...], 轨迹聚类的结果存储字典, key为聚类ID, value为类簇下的segment列表
        min_lines: int, 满足segment数的最小值
        min_dist: float, 生成的轨迹点之间的最小距离, 生成的轨迹点之间的距离不能太近的控制参数
    return
    ------
        Dict[int, List[Point, ...], ...], 每个类别下的代表性轨迹结果
    """
    representative_point = defaultdict(list)
    for i in cluster_segment.keys():
        cluster_size = len(cluster_segment.get(i))
        sort_point = []  # [Point, ...], size = cluster_size*2
        rep_point, zero_point = Point(0, 0, -1), Point(1, 0, -1)
        rdd_segments = sc.parallelize(cluster_segment.get(i))
        rep_point = rdd_segments.map(lambda seg: seg.end - seg.start).reduce(lambda vec1, vec2: vec1 + vec2)
        rep_point = rep_point / float(cluster_size)  # 对所有点的x, y求平局值

        cos_theta = rep_point.dot(zero_point) / rep_point.distance(Point(0, 0, -1))  # cos(theta)
        sin_theta = math.sqrt(1 - math.pow(cos_theta, 2))  # sin(theta)

        # 对某个i类别下的所有segment进行循环, 每个点进行坐标变换: X' = A * X => X = A^(-1) * X'
        #   |x'|      | cos(theta)   sin(theta) |    | x |
        #   |  |  =   |                         | *  |   |
        #   |y'|      |-sin(theta)   cos(theta) |    | y |

        rdd_segment_transformed = rdd_segments.map(lambda seg: segment_transformation(seg, sin_theta, cos_theta))
        cluster_segment[i] = rdd_segment_transformed.collect()
        sort_point = rdd_segment_transformed.flatMap(lambda seg: [seg.start, seg.end]).sortBy(lambda p: p.x).collect()
        for p in range(len(sort_point)):
            intersect_cnt = 0.0 + rdd_segment_transformed.map(lambda seg: calc_intersect_cnt(seg, p, sort_point)).reduce(lambda point1, point2: point1 + point2)
            # 分布式计算the average coordinate: avg_p and dist >= min_dist
            if intersect_cnt >= min_lines:
                start_y = Point(0, 0, -1) + rdd_segment_transformed.map(lambda seg: calc_start_y(seg, p, sort_point)).reduce(lambda point1, point2: point1 + point2)
                tmp_point = start_y / intersect_cnt
                # 坐标转换到原始的坐标系, 通过逆矩阵的方式进行矩阵的计算:https://www.shuxuele.com/algebra/matrix-inverse.html
                tmp = Point(tmp_point.x * cos_theta - sin_theta * tmp_point.y,
                            sin_theta * tmp_point.x + cos_theta * tmp_point.y, -1)
                _size = len(representative_point[i]) - 1
                if _size < 0 or (_size >= 0 and tmp.distance(representative_point[i][_size]) > min_dist):
                    representative_point[i].append(tmp)
    return representative_point