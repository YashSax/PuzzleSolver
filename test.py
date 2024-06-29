import cv2
import matplotlib.pyplot as plt
import numpy as np

unsolved = cv2.imread("./unsolved.png")
unsolved = cv2.cvtColor(unsolved, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(unsolved, 127, 255, cv2.THRESH_BINARY)

pieces = []
width = len(thresholded[0])
height = len(thresholded)
visited = [[False for _ in range(width)] for __ in range(height)]

def neighbors(i, j):
    n = []
    if i > 0:
        n.append([i - 1, j])
    if i < height - 1:
        n.append([i + 1, j])
    if j > 0:
        n.append([i, j - 1])
    if j < width - 1:
        n.append([i, j + 1])
    return n

def dfs(i, j):
    st = [[i, j]]
    while len(st) != 0:
        curr_i, curr_j = st.pop()
        visited[curr_i][curr_j] = True
        for n in neighbors(curr_i, curr_j):
            if thresholded[n[0]][n[1]] == 255 and not visited[n[0]][n[1]]:
                st.append(n)

for i in range(height):
    for j in range(width):
        if not visited[i][j] and thresholded[i][j] == 255:
            pieces.append((i, j))
            dfs(i, j)

print(pieces)