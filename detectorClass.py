import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

class SudokuDetector:
    
    def __init__(self, debug=False, size=400):
        self.debug = debug
        self.size = size
        
    def detect(self, image):
        img = image.copy()

        strategies = [
            self._strategy_blur,
            self._strategy_laplacian_kernel
        ]

        best_score = -1
        best_result = None
        best_points = None

        for i, strategy in enumerate(strategies):
            try:
                result, points, score = strategy(image.copy())
                if self.debug:
                    print(f"Strategy {i+1} score: {score}")
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title('Original')
                    plt.subplot(122)
                    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    plt.title(f'Strategy {i+1} (Score: {score:.2f})')
                    plt.show()
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_points = points

            except Exception as e:
                if self.debug:
                    print(f"Error in strategy {i}: {str(e)}")
                continue
        
        if best_score > 0:
            return best_result, best_points, True
        else:
            # Fallback to the original image with a message
            cv2.putText(img, "Detection failed", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img, None, False
        

    def _validate_grid(self, points, image_shape):
        if points is None or len(points) != 4:
            return 0
        print("Image shape:", image_shape)
        # Convert to numpy array if not already
        points = np.array(points)
        print(f"Points: {points}")
        
        # Calculate area - should be substantial portion of the image
        area = cv2.contourArea(points)
        print(f"Area: {area}")
        image_area = image_shape[0] * image_shape[1]
        print(f"Image Area: {image_area}")
        area_ratio = area / image_area
        print(f"Area Ratio: {area_ratio}")
        # Grid should be at least 10% of image but not more than 98%
        if area_ratio < 0.1 or area_ratio > 0.98:
            return 0
        
        # Check if shape is approximately square
        # Get bounding rect dimensions
        x, y, w, h = cv2.boundingRect(points)
        aspect_ratio = max(w/h, h/w)  # Will be 1 for a perfect square
        
        # Sudoku grid should be approximately square (allowing some perspective distortion)
        if aspect_ratio > 1.5:
            return 0
        
        # Calculate score based on how square it is and its size
        squareness_score = 1 - min(1, (aspect_ratio - 1))
        size_score = min(area_ratio * 2, 1)  # Favor larger grids up to 50% of image
        
        # Final score is weighted combination
        score = 0.6 * squareness_score + 0.4 * size_score
        
        return score

    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[2] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[3] = myPoints[np.argmax(diff)]
        return myPointsNew


    def biggestContour(self, contours, img):
        contour_images = []  # Store intermediate contour images
        biggest = np.array([])
        max_area = 0
        max_score = -1

        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                epsilon = 0.02
                approx = cv2.approxPolyDP(i, epsilon * peri, True)
                while len(approx) > 4 and epsilon < 0.2:
                    epsilon += 0.01
                    approx = cv2.approxPolyDP(i, epsilon * peri, True)
                imgBigContour = img.copy()
                if len(approx) == 4:
                    approx = self.reorder(approx)
                    score = self._validate_grid(approx, img.shape)
                    if score > max_score and area > max_area:
                        cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                        biggest = approx
                        max_area = area
                        max_score = score
                else:
                    cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                contour_images.append(imgBigContour)  # Save the contour image
        
        print(biggest, max_area, max_score)
        return biggest, max_area, contour_images, max_score

    def extract_grid(self, img, points):
        imgContours = img.copy()
        imgBlank = np.zeros((self.size, self.size, 3), np.uint8)

        if points.size != 0:
            points = self.reorder(points)
            cv2.drawContours(imgContours, points, -1, (0, 0, 255), 20)
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [self.size, 0], [self.size, self.size], [0, self.size]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (self.size, self.size))
            imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        else:
            imgWarpColored = imgBlank.copy()
        
        return imgWarpColored
    
    def _strategy_blur(self, img, kernel_size=17):
        if kernel_size < 1:  # Base case to prevent infinite recursion
            print("Kernel size too small to continue processing.")
            return np.zeros((self.size, self.size, 3), np.uint8), None, 0
        
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 0)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        imgDilation = cv2.dilate(imgThreshold, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(imgDilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for the largest contour that could be a sudoku grid
        biggest_contour, _, _, max_score = self.biggestContour(contours, img)
        
        # Draw the result
        imgWarpColored = self.extract_grid(img, biggest_contour)
        
        edges = cv2.Canny(imgWarpColored, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
        if lines is None:
            imgWarpColored, biggest_contour, max_score = self._strategy_blur(img, kernel_size - 2)

        return imgWarpColored, biggest_contour, max_score
    
    def _strategy_laplacian_kernel(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgLaplacian1 = cv2.Laplacian(imgGray, cv2.CV_64F)
        imgLaplacian2 = cv2.convertScaleAbs(imgLaplacian1)
        imgThreshold = cv2.adaptiveThreshold(imgLaplacian2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        kernel = np.ones((3, 3), np.uint8)
        imgOpening = cv2.morphologyEx(imgThreshold, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(imgOpening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        biggest_contour, _, _, max_score = self.biggestContour(contours, img)
        
        # Draw the result
        imgWarpColored = self.extract_grid(img, biggest_contour)
        
        return imgWarpColored, biggest_contour, max_score
    
    def process_dataset(self, folder_path, output_folder=None):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        stats = {
            'total': len(image_files),
            'success': 0,
            'failure': 0
        }
        
        # Process each image
        for i, filename in enumerate(image_files):
            try:
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Failed to read image: {filename}")
                    stats['failure'] += 1
                    continue
                
                # Process the image
                result, points, success = self.detect(image)
                
                if success:
                    stats['success'] += 1

                        
                    # Save or display results
                    if output_folder:
                        base_name = os.path.splitext(filename)[0]
                        
                        # Save the detection result
                        result_path = os.path.join(output_folder, f"{base_name}_detection.jpg")
                        cv2.imwrite(result_path, result)
                        
                        # Save the extracted grid if available
                        if result is not None:
                            grid_path = os.path.join(output_folder, f"{base_name}_grid.jpg")
                            cv2.imwrite(grid_path, result)
                    
                    if self.debug:
                        print(f"Successfully processed {filename} ({i+1}/{len(image_files)})")
                        
                else:
                    stats['failure'] += 1
                    if output_folder:
                        # Save the failed detection
                        fail_path = os.path.join(output_folder, f"failed_{filename}")
                        cv2.imwrite(fail_path, result)
                    
                    if self.debug:
                        print(f"Failed to process {filename} ({i+1}/{len(image_files)})")
                
            except Exception as e:
                stats['failure'] += 1
                print(f"Error processing {filename}: {str(e)}")
        
        # Print summary
        print(f"Processing complete: {stats['success']} successful, {stats['failure']} failed out of {stats['total']} images")
        
        return stats


# Example usage:
detector = SudokuDetector(debug=True)

# Process a single image
image = cv2.imread('changed_background/valid_grids/valid_grid_0_2025-04-28_12-55-46.png')
result, points, success = detector.detect(image)

if success and points is not None:
    grid = detector.extract_grid(image, points)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Detection')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    plt.title('Extracted Grid')
    plt.show()

# # Process an entire dataset
# stats = detector.process_dataset('path/to/dataset', 'path/to/output')