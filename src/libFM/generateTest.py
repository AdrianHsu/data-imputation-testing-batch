import sys
import random

path = sys.argv[1]
rowN = int(sys.argv[2])
colN = int(sys.argv[3])

with open(path, "w") as f:
	for i in range(rowN):
		for j in range(colN):
			f.write(str(i) +" " + str(j+rowN) + " " + str(random.randint(0,5)) + "\n")
