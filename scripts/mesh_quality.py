from dolfin import *
from dolfin import MeshFunction, cells


def print_mesh_quality_info(mesh):
    num_cells = mesh.num_cells()
    num_vertices = mesh.num_vertices()
    min_diameter = 2 * mesh.rmin()
    max_diameter = 2 * mesh.rmax()

    # Output the table
    print(f"{'Mesh Quality Information':^35}")
    print("-" * 35)
    print(f"{'Number of cells':<20} : {num_cells}")
    print(f"{'Number of vertices':<20} : {num_vertices}")
    print(f"{'Minimum cell diameter':<20} : {min_diameter:.6f}")
    print(f"{'Maximum cell diameter':<20} : {max_diameter:.6f}")
    return {
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "min_diameter": min_diameter,
        "max_diameter": max_diameter,
    }


def as_latex_table(num_cells, num_vertices, min_diameter, max_diameter):
    # Output the table
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\begin{tabular}{lr}")
    print(r"\hline")
    print(r"\textbf{Mesh Quality Information} & \\")
    print(r"\hline")
    print(rf"Number of cells & {num_cells} \\")
    print(rf"Number of vertices & {num_vertices} \\")
    print(rf"Minimum cell diameter & {min_diameter:.6f} \\")
    print(rf"Maximum cell diameter & {max_diameter:.kf} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Mesh Quality Metrics}")
    print(r"\label{table:mesh_quality}")
    print(r"\end{table}")


# Example usage
if __name__ == "__main__":
    # Create a test mesh (for example, a unit square mesh)
    mesh = UnitSquareMesh(8, 8)
    mesh_quality = print_mesh_quality_info(mesh)
    print()
    as_latex_table(**mesh_quality)
