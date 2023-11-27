import subprocess

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("threads", type=int)
    args = parser.parse_args()
    subprocess.run(f"mpirun -n {args.threads} python scripts/test_subprocess.py", shell=True)
