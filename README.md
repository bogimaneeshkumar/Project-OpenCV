# Project-Opencv

### PROJECT OUTLINE
 To detect the squares in an image and to overlap the aruco markers of specific ids to the respective coloured square boxes
 ### STEPS INVOLVED
 
 1. To resize the given image as to set it in the window size without changing the aspect ratio 
 2. Identifing the squares in the given image and extracted the location coordinates of the squares in the window and contours are also drawn
 3. Orientation,dimension[area] of the squares in the given image are being identified 
 4. Detecting the marker id of the aruco markers as per the rules to place  them in the respective colour square boxes the orientation, cropping ,background masking ,resizing has been done 
 5. The aruco markers are then overlapped on the square boxes as per the specified rules
    GREEN = MARKER ID 1
    ORANGE= MARKER ID 2
    BLACK = MARKER ID 3
    PINK-PINCH= MARKER ID 4
 6. The final overalapped image is again resized to its initial original dimensions
