def edit_distance(seq1, seq2): 
    col = len(seq2) + 1
    row = len(seq1) + 1

    change = [0,0,0]

    matrix = [[0] * col for _ in range(row)]

    for i in range(col):
        matrix[0][i] = i
    for j in range(row):
        matrix[j][0] = j

    for i in range(1, row) : 
        for j in range(1,col) : 
            if seq1[i-1] == seq2[j-1] : 
                change[0] = matrix[i-1][j-1]
            else : 
                change[0] = matrix[i-1][j-1]+1
            change[1] = matrix[i][j-1]+1
            change[2] = matrix[i-1][j]+1
            print(change)
            matrix[i][j] = min(change)
    print(matrix)
    return matrix[row-1][col-1]

print(edit_distance("ABC", "ABD"))






