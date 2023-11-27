import dolfin as df


if __name__ == "__main__":
    print(df.MPI.comm_world.rank, "/",  df.MPI.comm_world.size)
