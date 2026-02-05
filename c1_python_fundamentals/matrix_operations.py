def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Incompatible dimensions")

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def spiral_traverse(matrix):
    result = []
    top, bottom = 0, len(matrix)-1
    left, right = 0, len(matrix[0])-1

    while top <= bottom and left <= right:
        for i in range(left, right+1):
            result.append(matrix[top][i])
        top += 1

        for i in range(top, bottom+1):
            result.append(matrix[i][right])
        right -= 1

        for i in range(right, left-1, -1):
            result.append(matrix[bottom][i])
        bottom -= 1

        for i in range(bottom, top-1, -1):
            result.append(matrix[i][left])
        left += 1

    return result

if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]

    print("Matrix Multiplication Result:")
    print(matrix_multiply(A, B))

    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    print("Spiral Traversal:")
    print(spiral_traverse(matrix))
