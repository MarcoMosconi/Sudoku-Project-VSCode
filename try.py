import numpy as np
import cv2
import matplotlib.pyplot as plt 

def preProcess(img, kernel_size):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 0)    
    # # Apply Laplacian edge detection
    # imgLaplacian1 = cv2.Laplacian(imgBlur, cv2.CV_64F)
    # imgLaplacian2 = cv2.convertScaleAbs(imgLaplacian1)  # Convert to 8-bit image
    # kernel = np.ones((5, 5), np.uint8)
    # imgLaplacian3 = cv2.dilate(imgLaplacian2, kernel, iterations=1)

    imgLaplacian1 = cv2.Laplacian(imgGray, cv2.CV_64F)
    imgLaplacian2 = cv2.convertScaleAbs(imgLaplacian1)

    imgThresh = cv2.adaptiveThreshold(imgLaplacian2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 2)
    kernel = np.ones((3, 3), np.uint8)
    imgOpening = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Display all images together
    images = [imgGray, imgLaplacian2, imgThresh, imgOpening]
    titles = ["Grayscale Image", "Laplacian Image", "Threshold Image", "Opening Image"]

    plt.figure(figsize=(10, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    return imgOpening

def biggestContour(contours, img):
    contour_images = []  # Store intermediate contour images
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

            if area > max_area and len(approx) == 4:
                if approx.size != 0:
                    approx = reorder(approx)
                    cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                biggest = approx
                max_area = area
            else:
                cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)

            contour_images.append(imgBigContour)  # Save the contour image

    return biggest, max_area, contour_images

def reorder(myPoints):
  myPoints = myPoints.reshape((4, 2))
  myPointsNew = np.zeros((4, 1, 2), np.int32)
  add = myPoints.sum(1)
  myPointsNew[0] = myPoints[np.argmin(add)]
  myPointsNew[2] = myPoints[np.argmax(add)]
  diff = np.diff(myPoints, axis=1)
  myPointsNew[1] = myPoints[np.argmin(diff)]
  myPointsNew[3] = myPoints[np.argmax(diff)]
  return myPointsNew

def defineGrid(image_path, size, kernel_size):
    img = cv2.imread(image_path)
    imgBlank = np.zeros((size, size, 3), np.uint8)

    # Update to unpack the Laplacian image
    imgLaplacian = preProcess(img, kernel_size)

    imgContours = img.copy()
    # Use Laplacian image for contour detection
    contours, hierarchy = cv2.findContours(imgLaplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 
    # cv2.imshow("Contours", imgContours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    biggest, maxArea, contour_images = biggestContour(contours, img)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgContours, biggest, -1, (0, 0, 255), 20)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (size, size))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        print("Va")
    else:
        print("Nun va")
        imgWarpColored = imgBlank.copy()

    # Plot only the final chosen image
    plt.figure(figsize=(6, 6))
    plt.imshow(imgWarpColored, cmap='gray')
    plt.title("Final Image with Detected Grid")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return imgWarpColored

defineGrid("sample/sample_valid_grids/valid_grid_5_2025-04-28_13-07-23.png", 400, 3)
