import sys, yaml, os.path
project = os.path.expanduser(
        yaml.safe_load(open(sys.argv[1]))["project"])
print("../models/" + project)