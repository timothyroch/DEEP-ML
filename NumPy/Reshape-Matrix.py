import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    if len(a) == 0 or len(a[0]) == 0:
        return []

    row = new_shape[0]
    col = new_shape[1]

    # --- Manual version ---
    # nums = []
    # for i in range(len(a)):
    #     for j in range(len(a[0])):
    #         nums.append(a[i][j])
    # 
    # if len(nums) != row * col:
    #     return []
    # 
    # tt = []
    # for i in range(row):
    #     curr = nums[i * col : (i + 1) * col]
    #     tt.append(curr)

    arr = np.array(a)
    
    if arr.size != row * col:
        return []

    return arr.reshape(new_shape).tolist()
