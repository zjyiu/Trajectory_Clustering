import math

from point import Point, _point2line_distance, show_point


class Segment(object):
    """将一个segment进行封装, 进行距离(垂直距离, 长度距离, 角度距离)的计算, 设置segment的cluster的ID等, 在使用时需区分长短segment,两者的调用方式不同
    method
    ------
        perpendicular_distance: 计算垂直距离, longer_segment.perpendicular_distance(short_segment)
        parallel_distance: 计算segment长度的相似度, longer_segment.parallel_distance(short_segment)
        angle_distance: 计算两个segment的角度相似性, longer_segment.angle_distance(short_segment)
    """
    eps = 1e-12

    def __init__(self, start_point, end_point, traj_id = None, cluster_id = -1):
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id
        self.cluster_id = cluster_id

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_traj_id(self):
        return self.traj_id

    def get_cluster_id(self):
        return self.cluster_id

    def set_cluster(self, cluster_id):
        self.cluster_id = cluster_id

    def pair(self) :
        return self.start, self.end

    @property
    def length(self):
        return self.end.distance(self.start)

    def perpendicular_distance(self, other):
        """计算两个segment之间起始点的垂直距离距离, 参考论文中的公式Formula(1); 必须Segment为short的line segment."""
        if self.length==0:# 如果线段长度为0，以点到线段的距离作为结果
            return _point2line_distance(self.start.as_array(),other.start.as_array(),other.end.as_array())
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)

    def parallel_distance(self, other):
        """计算两个segment之间的长度距离, 参考论文中的公式Formula(2),Segment必须为short的line segment."""
        if self.length==0:# 如果线段长度为0，直接以另一条线段的长度为结果
            return other.length
        l1 = self.start.distance(self._projection_point(other, typed='start'))
        l2 = self.end.distance(self._projection_point(other, typed='end'))
        return min(l1, l2)

    def angle_distance(self, other):
        """计算两个segment之间的角度距离, 参考论文中的公式Formula(3),Segment必须为short的line segment."""
        if self.length==0:# 如果线段长度为0，直接以另一条线段的长度为结果
            return other.length
        self_vector = self.end - self.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        # 当两个点重合时, 计算点到直线的距离即可
        if self_dist < self.eps:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist < self.eps:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        cos_theta = self_vector.dot(other.end - other.start) / (self.end.distance(self.start) * other.end.distance(other.start))
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length

    def _projection_point(self, other, typed="e"):
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        u = tmp.dot(self.end-self.start) / math.pow(self.end.distance(self.start), 2)
        return self.start + (self.end-self.start) * u

    def get_all_distance(self, seg):
        res = self.angle_distance(seg)
        # 起始点不能为同一个点
        if str(self.start) != str(self.end):
            res += self.parallel_distance(seg)
        # 不能为同一轨迹
        if self.traj_id != seg.traj_id:
            res += self.perpendicular_distance(seg)
        return res


def compare(segment_a, segment_b):
    """对两个segment进行对比, 返回:(longer_segment, shorter_segment)"""
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)


def show_seg(seg):
    show_point(seg.get_start())
    show_point(seg.get_end())
    if seg.get_traj_id() is None:
        traj_id = -1
    else:
        traj_id = seg.get_traj_id()
    if seg.get_cluster_id() is None:
        cluster_id = -1
    else:
        cluster_id = seg.get_cluster_id()
    print ", %d, %d\n" % (traj_id, cluster_id)
