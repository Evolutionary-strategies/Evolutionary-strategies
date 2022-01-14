import sys
import numpy as np
print(np.frombuffer(sys.argv[1].encode("utf8", "surrogateescape")))