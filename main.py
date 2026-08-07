def li(list):
    return list[1][2]
lk = [[1, 2, [3, 4, 5]], [6, 7, [8, 9, 10]], [11, 12, [13, 14, 15]]]
sorted(lk, key=li)