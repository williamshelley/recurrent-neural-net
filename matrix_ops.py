

def mat_vec_multiply(matrix, vector):
  result = [0.0 for _ in matrix]

  print(matrix, vector)
    
  for r in range(len(matrix)):
    row = matrix[r]

    if len(row) != len(vector):
      raise Exception("Invalid matrix * vector dimensions")

    result[r] += sum([vector[i] * row[i] for i in range(len(row))])
  return result

def transpose(matrix):
  M = len(matrix)
  N = len(matrix[0])
  return [[matrix[r][c] for r in range(M)] for c in range(N)]

def mat_scalar_op(matrix, scalar, operation):
  result = [[x for x in row] for row in matrix]
  for r in range(len(result)):
    for c in range(len(result[r])):
      result[r][c] = operation(result[r][c], scalar)
  return result

def vec_vec_op(A, B, operation):
  if len(A) != len(B):
    raise Exception("Vector A and Vector B must be same size")

  return [operation(A[i], B[i]) for i in range(len(A))]

def vec_scalar_op(vector, scalar, operation):
  if type(vector) != list:
    raise Exception("Vector must be list")

  return [operation(e, scalar) for e in vector]
