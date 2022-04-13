import cv2
import numpy as np

# Display image
def display(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Remove shadow from the imaage
def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
        
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov

# Remove noise from the imaage
def remove_noise(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:   #filter small dotted regions
            img2[labels == i + 1] = 255
    return img2

# To break characters and returns a list of images of characters - 
# We are traversing across the image's width and checking every strip of width 1 pixel and 
# height equal to the image height. If the strip contains no white pixel, we break the image at that point.
def segmentation(img, original):
    segmented_img = []
    h, w = img.shape
    prev = 0
    for i in range(w):
        roi = img[:,i:i+1]
        coord = np.where(roi == [255])
        if len(coord[0]) == 0 and len(coord[1]) == 0:
            roi2 = img[:,prev:i+1]
            prev = i
            coord2 = np.where(roi2 == [255])
            if len(coord2[0]) == 0 and len(coord2[1]) == 0:
                continue
            else:
                cv2.line(original, (i,0),(i,h), (255,0,0),10)
                segmented_img.append(roi2)
                    
                
    return segmented_img

# Filters segmented images and returns finals list of images of characters - 
def purify(segmented_img):
    largest_area = 0
    for img in segmented_img:
        h,w = img.shape
        largest_area = max(largest_area, h*w)

    final_imgs =[]
    for img in segmented_img:
        h,w = img.shape
        if(h*w >= largest_area/15):
            final_imgs.append(img)
    return final_imgs


# Crop the extra area from final list of images and returns it
def crop(segmented_img, img):
    h, w = img.shape
    # CROPPING
    for x in range(len(segmented_img)):
        
        coord = np.where(segmented_img[x] == [255])
        sorted_y_coord = np.sort(coord[0])
        sorted_x_coord = np.sort(coord[1])
        min_y = sorted_y_coord[0]
        min_x = sorted_x_coord[0]
        max_y = sorted_y_coord[-1]
        max_x = sorted_x_coord[-1]
        dx = 0
        dy = 0
        if min_x >= 10 and w - max_x >= 10:
            dx = 10
        else:
            dx = min(min_x, w - max_x)

        if min_y >= 10 and h - max_y >= 10:
            dy = 10
        else:
            dy = min(min_y, h - max_y)

        
        segmented_img[x] = segmented_img[x][int(min_y - dy): int(max_y + dy), int(min_x - dx): int(max_x + dx)]

        # RESIZING IMAGE
        segmented_img[x] = cv2.resize(segmented_img[x], (28, 28), interpolation=cv2.INTER_AREA)

    return segmented_img

