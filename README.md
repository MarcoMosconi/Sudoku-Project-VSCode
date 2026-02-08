This project is designed to build a dataset of raw Sudoku puzzle images, processing those images to extract Sudoku grids and train several models to classify them as either solvable or not, combining neural and symbolic reasoning. 
It includes utility scripts for dataset creation and validation, along with datasets and transformed images that are then used for training and evaluation of the designed models.
Both a simple CNN and a CNN using symbolic reasoning were Sudoku rules are explicitely defined using the Scallopy library are tested.  
Structure: 
### Project Structure
```
Sudoku-Project-VSCode/
├── .vscode/                     # VS Code workspace configs 
├── grid_extraction/             # extraction of Sudoku grids from raw images 
├── image_generation/            # generation of synthetic images
├── models/                      # models definitions and training code
├── sample/                      # sample input files or examples 
├── sudoku_grids_dataset/        # dataset of extracted grids 
├── sudoku_photos_dataset/       # dataset of raw photos of Sudoku puzzles 
├── transformed_images/          # geometric transformated Sudoku grids before generation of synthetic background
├── useless_stuff/               # Misc / unused files
├── create_dataset.py            # Script to build datasets 
├── validate.py                  # Script to validate grid data
├── .gitignore                   # Git ignore rules :contentReference[oaicite:12]{index=12}
└── README.md                   # This file
```
