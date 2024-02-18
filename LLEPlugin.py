from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding

class LLEPlugin:
    def input(self, inputfile):
       infile = open(inputfile, 'r')
       self.X = []
       for line in infile:
          row = line.strip().split(',')
          for i in range(len(row)):
            row[i] = float(row[i])
          self.X.append(row)

    def run(self):
       embedding = LocallyLinearEmbedding(n_components=2)
       self.X_transformed = embedding.fit_transform(self.X[:100])
    def output(self, outputfile):
       print(self.X_transformed)
