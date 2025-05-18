import numpy as np
import cv2
import matplotlib.pyplot as plt  

def preProcess(img, min_line_length=1, max_line_gap=10, size=400):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (0, 0), 0)
    edges = cv2.Canny(imgGray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        print("No lines detected using HoughLinesP.")
        return np.zeros((size, size, 3), np.uint8), None

        # Draw detected lines on a blank image
    line_img = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Display all images together
    images = [imgGray, edges, line_img]
    titles = ["Grayscale Image","Canny Image", "Line Image"]

    plt.figure(figsize=(10, 8))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return line_img

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


def biggestContour(contours, img):
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
                approx = reorder(approx)
                cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
                biggest = approx
                max_area = area
            else:
                cv2.drawContours(imgBigContour, approx, -1, (0, 0, 255), 20)
            contour_images.append(imgBigContour)  
    print(biggest, max_area)
    return biggest, max_area, contour_images

def defineGrid(image_path, min_line_length=50, max_line_gap=10, size=400):
        img = cv2.imread(image_path)

        lineImage = preProcess(img, min_line_length, max_line_gap, size)

        imgBlank = np.zeros((size, size, 3), np.uint8)
        imgContours = img.copy()

        # Find contours from the line image
        contours, _ = cv2.findContours(cv2.cvtColor(lineImage, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        biggest, _, _ = biggestContour(contours, img)

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
        
        plt.figure(figsize=(6, 6))
        plt.imshow(imgWarpColored, cmap='gray')
        plt.title("Final Image with Detected Grid")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


        return imgWarpColored, biggest

defineGrid("sample/sample_valid_grids/valid_grid_5_2025-04-28_13-07-23.png")
