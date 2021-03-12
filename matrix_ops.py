

def mat_vector_multiply(matrix, vector):
  result = [0.0 for _ in matrix]
    
  for r in range(len(matrix)):
    row = matrix[r]

    if len(row) != len(vector):
      raise Exception("Invalid matrix * vector dimensions")

    result[r] += sum([vector[i] * row[i] for i in range(len(row))])
  return result

def mat_scalar_op(matrix, scalar, operation):
  result = [[x for x in row] for row in matrix]
  for r in range(len(result)):
    for c in range(len(result[r])):
      result[r][c] = operation(result[r][c], scalar)
  return result