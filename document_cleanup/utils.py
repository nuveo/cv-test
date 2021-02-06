def sortPointsClockwise(points):
        """
        Sort points in clockwise direction
        """

        def less(center, point_1, point_2):
            """
            Returns if point 1 is to the right of point 2 in relation to the center
            """
            if point_1[0] - center[0] >= 0 and point_2[0] - center[0] < 0:
                return True
            if point_1[0] - center[0] < 0 and point_2[0] - center[0] >= 0:
                return False
            if point_1[0] - center[0] == 0 and point_2[0] - center[0] == 0:
                if point_1[1] - center[1] >= 0 or point_2[1] - center[1] >= 0:
                    return point_1[1] > point_2[1]
                return point_2[1] > point_1[1]

            det = (point_1[0] - center[0]) * (point_2[1] - center[1]) - \
                (point_2[0] - center[0]) * (point_1[1] - center[1])
            if det < 0:
                return True
            if det > 0:
                return False

            d1 = (point_1[0] - center[0]) * (point_1[0] - center[0]) + \
                (point_1[1] - center[1]) * (point_1[1] - center[1])
            d2 = (point_2[0] - center[0]) * (point_2[0] - center[0]) + \
                (point_2[1] - center[1]) * (point_2[1] - center[1])
            return d1 > d2

        def getCenter(points):
            center = [0, 0]

            for point in points:
                center[0] += point[0]
                center[1] += point[1]

            center[0] = center[0] / len(points)
            center[1] = center[1] / len(points)

            return center

        center = getCenter(points)

        # bubble sort
        for i in range(len(points) - 1, 0, -1):
            for j in range(i):
                if less(center, points[j], points[j + 1]):
                    swap = points[j].copy()
                    points[j] = points[j + 1]
                    points[j + 1] = swap

        # print(points)
        return points
