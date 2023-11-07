import cv2
import numpy as np
import random

# Read the image
image = cv2.imread('clock_on_a_beach.png')

# Define the size of the squares in the grid
square_size = 50  # size of the square in pixels

# Create a mask with the same dimensions as the image, initialized to zeros (black)
mask = np.zeros_like(image)

# Loop through the grid
for i in range(0, image.shape[0], square_size):
    for j in range(0, image.shape[1], square_size):
        # Randomly decide whether to mask this square
        if random.choice([True, False]):
            # Set all pixels in this square to zero (black)
            mask[i:i+square_size, j:j+square_size] = 0
        else:
            # Otherwise, copy the pixels from the original image
            mask[i:i+square_size, j:j+square_size] = image[i:i+square_size, j:j+square_size]

# Save or show the masked image
cv2.imwrite('masked_image.jpg', mask)
# cv2.imshow('Masked Image', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()