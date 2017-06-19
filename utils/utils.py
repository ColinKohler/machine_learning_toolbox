

# Load the class encoding
def loadClassEncoding(path):
    one_hot_encoding = dict()
    with open(path, 'r') as f:
        for line in f:
            cls, num = line.split(' ', 1)
            one_hot_encoding[cls] = int(num)
    return one_hot_encoding
