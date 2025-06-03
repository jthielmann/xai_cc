import sys, yaml, os.path
print(os.path.expanduser(
        yaml.safe_load(open(sys.argv[1]))["parameters"]["out_path"]["value"]))