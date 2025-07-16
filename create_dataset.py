import os
import shutil

# def copy_files(src_dir, dest_dir, exclude_dir):
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
#     for filename in os.listdir(src_dir):
#         if filename not in os.listdir(exclude_dir) and filename not in os.listdir(dest_dir):
#             source_path = os.path.join(src_dir, filename)
#             destination_path = os.path.join(dest_dir, filename)
#             shutil.copy(source_path, destination_path)
    
# copy_files("grid_extraction/extracted_grids/photos_wrong_grids", "sudoku_grids_dataset/invalid", "grid_extraction/badly_extracted_grids/wrong_photos_wrong_grids")
# copy_files("grid_extraction/extracted_grids/photos_valid_grids", "sudoku_grids_dataset/valid", "grid_extraction/badly_extracted_grids/wrong_photos_valid_grids")
# copy_files("grid_extraction/extracted_grids/aws_wrong_grids", "sudoku_grids_dataset/invalid", "grid_extraction/badly_extracted_grids/wrong_aws_wrong_grids")
# copy_files("grid_extraction/extracted_grids/aws_valid_grids", "sudoku_grids_dataset/valid", "grid_extraction/badly_extracted_grids/wrong_aws_valid_grids")

i = 0
j = 0
for filename in os.listdir("sudoku_grids_dataset/valid"):
    j += 1
for filename in os.listdir("sudoku_grids_dataset/invalid"):
    i += 1

print("Number of aws valid grids: ", j)
print("Number of aws invalid grids: ", i)
print("Total number of grids: ", i + j)