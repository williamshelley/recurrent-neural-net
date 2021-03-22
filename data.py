ZERO = [0]
ONE = [1]
T1 = [0,0]
T2 = [0,1]
T3 = [1,0]
T4 = [1,1]
AND_FILE = "AND.json"
OR_FILE = "OR.json"
XOR_FILE = "XOR.json"


class Data:
  @staticmethod
  def AND():
    global ZERO, ONE, T1, T2, T3, T4, AND_FILE
    return ([[T1, ZERO], [T2, ZERO], [T3, ZERO], [T4, ONE]], AND_FILE)

  @staticmethod
  def OR():
    global ZERO, ONE, T1, T2, T3, T4, OR_FILE
    return ([[T1, ZERO], [T2, ONE], [T3, ONE], [T4, ONE]], OR_FILE)

  @staticmethod
  def XOR():
    global ZERO, ONE, T1, T2, T3, T4, XOR_FILE
    return ([[T1, ZERO], [T2, ONE], [T3, ONE], [T4, ZERO]], XOR_FILE)