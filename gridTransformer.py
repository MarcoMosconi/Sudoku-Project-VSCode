import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import requests
from sudokuCorruptor import generate_invalid_sudoku
import time

def create_sudoku_image(grid, cell_size=50):
    """Convert a sudoku grid (2D array) into an image."""
    # Determine grid dimensions
    if isinstance(grid, list):
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
    else:
        # Handle string input or other formats
        # This is placeholder - you would implement parsing logic
        rows, cols = 9, 9

    # Create a white image
    img_height = rows * cell_size
    img_width = cols * cell_size
    image = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Draw grid lines
    for i in range(rows + 1):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(image, (0, i * cell_size), (img_width, i * cell_size), 0, thickness)

    for i in range(cols + 1):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(image, (i * cell_size, 0), (i * cell_size, img_height), 0, thickness)

    for i in range(rows):
        for j in range(cols):
            if isinstance(grid, list) and grid[i][j] != 0:
                font = random.choice([cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX])
                font_scale = 0.8 if font == cv2.FONT_HERSHEY_DUPLEX else random.uniform(0.6, 0.8)
                font_thickness = 1 if font == cv2.FONT_HERSHEY_DUPLEX else random.randint(1, 2)
                # Calculate the position for the number
                text = str(grid[i][j])
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                # Randomly adjust the position slightly
                if font == cv2.FONT_HERSHEY_SCRIPT_COMPLEX:
                    text_x += random.randint(-10, 10)
                    text_y += random.randint(-10, 10)
                # Put the text in the image
                cv2.putText(image, text, (text_x, text_y), font, font_scale, 0, font_thickness)

    return image

def apply_perspective_transform(image, strength):
    """Apply a random perspective transform to the image."""

    h, w = image.shape[:2]

    # Define the four corners of the image
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Define how much distortion to apply
    distortion = int(min(h, w) * strength)

    # Create random offsets for each corner
    offset_range = (-distortion, distortion)
    new_corners = np.float32([
        [random.randint(*offset_range), random.randint(*offset_range)],  # top-left
        [w - random.randint(*offset_range), random.randint(*offset_range)],  # top-right
        [w - random.randint(*offset_range), h - random.randint(*offset_range)],  # bottom-right
        [random.randint(*offset_range), h - random.randint(*offset_range)]   # bottom-left
    ])

    # Add the offsets to the original corners
    new_corners = corners + new_corners

    # Ensure corners are within image bounds
    new_corners = np.clip(new_corners, 0, max(h, w))

    reduction_matrix = np.float32([
                                [50, 50],
                                [-50, 50],
                                [-50, -50],
                                [50, -50]])

    new_corners += reduction_matrix

    # print(new_corners)

    tuple_corners = [(int(corner[0]), int(corner[1])) for corner in new_corners]

    # print(tuple_corners)
    # print(tuple_corners[0])

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, new_corners)

    # Apply the transform
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Ensure grid edges are defined
    cv2.line(warped, tuple_corners[0], tuple_corners[1], 0, 2)  # Top edge
    cv2.line(warped, tuple_corners[0], tuple_corners[3], 0, 2)  # Left edge
    cv2.line(warped, tuple_corners[1], tuple_corners[2], 0, 2)  # Right edge
    cv2.line(warped, tuple_corners[2], tuple_corners[3], 0, 2)  # Bottom edge

    return warped

def add_photo_effects(image, shadow_strength, noise_level):
    """Add realistic photo effects like shadows, grain and lighting."""
    # Convert to 3-channel if it's grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    # Add vignette effect (darker edges)
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2

    # Create circular mask
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist_from_center / max_dist * shadow_strength)
    vignette = np.clip(vignette, 0, 1)

    # Apply vignette
    for i in range(3):
        result[:, :, i] *= vignette

    # Add noise (grain)
    noise = np.random.randn(h, w, 3) * noise_level
    result += noise

    # Clip values to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Optional: Add slight blur to simulate camera focus
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result

def process_sudoku_grid(grid, output_dir, grid_name,
                        perspective_strength=0.15,
                        shadow_strength=0.3,
                        noise_level=7,
                        ):
    """Process a single sudoku grid with all effects."""
    os.makedirs(output_dir, exist_ok=True)


    # Create basic sudoku image
    sudoku_img = create_sudoku_image(grid)

    # Apply perspective transform
    warped = apply_perspective_transform(sudoku_img, strength=perspective_strength)

    # Add photo effects
    final = add_photo_effects(warped, shadow_strength, noise_level)

    # Save the result
    output_path = os.path.join(output_dir, grid_name)
    cv2.imwrite(output_path, final)

    return final

def get_grid(output_dir, grid_name):
    """Fetch a Sudoku grid from the API if the file does not already exist."""
    output_path = os.path.join(output_dir, grid_name)
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping API request")
        return None

    time.sleep(random.randint(5, 9))
    response = requests.get('https://sudoku-api.vercel.app/api/dosuku?query={newboard(limit:1){grids{value, solution}}}')
    data = response.json()
    sudoku = data['newboard']['grids'][0]['value']
    solution = data['newboard']['grids'][0]['solution']
    grid = random.choice([sudoku, solution])
    return grid

for i in range(300):
    # Check and fetch valid grid
    grid_name = f"valid_grid_{i}.png"
    grid = get_grid("transformed_images/valid_grids", grid_name)
    if grid is not None:
        result = process_sudoku_grid(grid, "transformed_images/valid_grids", grid_name)
        print(f"Generated valid grid {i}")

    # Check and fetch corrupted grid
    grid_name = f"wrong_grid_{i}.png"
    grid = get_grid("transformed_images/wrong_grids", grid_name)
    if grid is not None:
        corrupted_grid = generate_invalid_sudoku(grid)
        result = process_sudoku_grid(corrupted_grid, "transformed_images/wrong_grids", grid_name)
        print(f"Generated wrong grid {i}")


