import nbformat

path = "models/CNN_combined (1).ipynb"

nb = nbformat.read(path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "execution_count" not in cell:
        cell.execution_count = None

nbformat.write(nb, path)
print("Notebook fixed.")

