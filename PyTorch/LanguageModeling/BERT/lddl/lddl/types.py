class File:

  def __init__(self, path, num_samples):
    self.path = path
    self.num_samples = num_samples

  def __repr__(self):
    return 'File(path={}, num_samples={})'.format(self.path, self.num_samples)
