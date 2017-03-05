import numpy as np

DEFAULT_FILE_PATH = "../data_VisualQA/cnn.txt"

def loadImgVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained CNN vectors"""
    imgVectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            imgVectors[tokens[token]] = np.asarray(data)
    return imgVectors