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
            self._strategy_laplacian_kernel,
            self._strategy_hough_lines_p  # New strategy added
        ]

        best_score = -1
        best_result = None
        best_points = None

        for i, strategy in enumerate(strategies):
            try:
                result, points = strategy(image.copy())
                score = self._evaluate_grid_quality(result)
                    
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
    
    def group_lines(self, lines, threshold=20):
        grouped = []
        for line in lines:
            if not grouped or abs(line - grouped[-1]) > threshold:
                grouped.append(line)
        return grouped

    def _evaluate_grid_quality(self, extracted_grid):
        try:
            grid = extracted_grid.copy()
            edges = cv2.Canny(grid, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            horizontal_lines = []
            vertical_lines = []
            for line in lines:
                rho, theta = line[0]
                if (theta < 0.26) or (theta > 2.88):  
                    vertical_lines.append(rho)
                elif (theta > 1.44) and (theta < 1.70):  
                    horizontal_lines.append(rho)
            horizontal_lines.sort()
            vertical_lines.sort()

            horizontal_lines = self.group_lines(horizontal_lines)
            vertical_lines = self.group_lines(vertical_lines)

            # Plot the image with detected lines
            # if self.debug:
            #     img_with_lines = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            #     for rho in horizontal_lines:
            #         a, b = np.cos(np.pi / 2), np.sin(np.pi / 2)
            #         x0, y0 = a * rho, b * rho
            #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            #         cv2.line(img_with_lines, pt1, pt2, (0, 255, 0), 2)
            #     for rho in vertical_lines:
            #         a, b = np.cos(0), np.sin(0)
            #         x0, y0 = a * rho, b * rho
            #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            #         cv2.line(img_with_lines, pt1, pt2, (255, 0, 0), 2)
            #     plt.figure(figsize=(10, 5))
            #     plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
            #     plt.title('Detected Lines')
            #     plt.show()
            
            h_position_score = 0.0
            v_position_score = 0.0
            
            if len(horizontal_lines) >= 7:
                h_position_score = (1 - abs(horizontal_lines[0]) / grid.shape[0]) + (1 - abs(horizontal_lines[-1] - grid.shape[0]) / grid.shape[0])
                h_position_score = max(0, h_position_score / 2)  

            if len(vertical_lines) >= 7:
                v_position_score = (1 - abs(vertical_lines[0]) / grid.shape[1]) + (1 - abs(vertical_lines[-1] - grid.shape[1]) / grid.shape[1])
                v_position_score = max(0, v_position_score / 2)  

            h_score = min(len(horizontal_lines) / 9, 1.0) * 0.8 + h_position_score * 0.2
            v_score = min(len(vertical_lines) / 9, 1.0) * 0.8 + v_position_score * 0.2
            
            final_score = 0.5 * h_score + 0.5 * v_score
            
            if self.debug:
                print(f"Horizontal lines: {len(horizontal_lines)}, Vertical lines: {len(vertical_lines)}")
                print(f"Horizontal valid: {h_score}, Vertical valid: {v_score}")
                print(f"Final grid quality score: {final_score:.2f}")
            
            return final_score
        
        except Exception as e:
            if self.debug:
                print(f"Error evaluating grid quality: {str(e)}")
            return 0.0

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
        contour_images = []  
        biggest = np.array([])
        max_area = 0

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

                if len(approx) == 4 and area > max_area:
                    approx = self.reorder(approx)
                    cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                    biggest = approx
                    max_area = area
                else:
                    cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                contour_images.append(imgBigContour)  
        print(biggest, max_area)
        return biggest, max_area, contour_images

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
        if (kernel_size < 1):  # Base case to prevent infinite recursion
            print("Kernel size too small to continue processing.")
            return np.zeros((self.size, self.size, 3), np.uint8), None
        
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 0)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        imgDilation = cv2.dilate(imgThreshold, kernel, iterations=1)
        
        contours, _ = cv2.findContours(imgDilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        biggest_contour, _, _ = self.biggestContour(contours, img)
        
        imgWarpColored = self.extract_grid(img, biggest_contour)
        
        edges = cv2.Canny(imgWarpColored, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
        if lines is None:
            imgWarpColored, biggest_contour = self._strategy_blur(img, kernel_size - 2)

        return imgWarpColored, biggest_contour
    
    def _strategy_laplacian_kernel(self, img, c=3):
        if c < 1:  
            print("Kernel size too small to continue processing.")
            return np.zeros((self.size, self.size, 3), np.uint8), None
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgLaplacian1 = cv2.Laplacian(imgGray, cv2.CV_64F)
        imgLaplacian2 = cv2.convertScaleAbs(imgLaplacian1)
        imgThreshold = cv2.adaptiveThreshold(imgLaplacian2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, c)
        kernel = np.ones((3, 3), np.uint8)
        imgOpening = cv2.morphologyEx(imgThreshold, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(imgOpening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        biggest_contour, _, _ = self.biggestContour(contours, img)
        
        imgWarpColored = self.extract_grid(img, biggest_contour)

        edges = cv2.Canny(imgWarpColored, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
        if lines is None:
            imgWarpColored, biggest_contour = self._strategy_laplacian_kernel(img, c - 1)
        
        return imgWarpColored, biggest_contour
    
    def _strategy_hough_lines_p(self, img, min_line_length=50, max_line_gap=10):
        """
        Strategy using cv2.HoughLinesP for grid detection.
        """
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(imgGray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

        if lines is None:
            if self.debug:
                print("No lines detected using HoughLinesP.")
            return np.zeros((self.size, self.size, 3), np.uint8), None

        # Draw detected lines on a blank image
        line_img = np.zeros_like(img)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Find contours from the line image
        contours, _ = cv2.findContours(cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour, _, _ = self.biggestContour(contours, img)

        imgWarpColored = self.extract_grid(img, biggest_contour)

        return imgWarpColored, biggest_contour
    
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
        
        for i, filename in enumerate(image_files):
            try:
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Failed to read image: {filename}")
                    stats['failure'] += 1
                    continue
                
                result, points, success = self.detect(image)
                
                if success:
                    stats['success'] += 1  
                    if output_folder:
                        base_name = os.path.splitext(filename)[0]
                        result_path = os.path.join(output_folder, f"{base_name}_detection.jpg")
                        cv2.imwrite(result_path, result)
                    
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

# # Process a single image
# image = cv2.imread('sample/sample_valid_grids/valid_grid_5_2025-04-28_13-07-23.png')
# result, points, success = detector.detect(image)

# if success and points is not None:
#     plt.figure(figsize=(10, 5))
#     plt.subplot(121)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title('Original')
#     plt.subplot(122)
#     plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
#     plt.title('Extracted Grid')
#     plt.show()

# Process an entire dataset
stats = detector.process_dataset('sample/sample_valid_grids', 'sample/sample_extracted_valid_grids')