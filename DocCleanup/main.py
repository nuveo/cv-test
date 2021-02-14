import sys, glob

import PreProcessing as pp


if __name__ == "__main__":

    image_list = glob.glob(sys.argv[1] + "/*.png")

    
    denoysed_images = pp.median_filter(image_list)
    #adjusted = brightness_and_contrast(denoysed_images, 0, 2)
    binarized_images = pp.binarize(denoysed_images)
    #Binarization is benefic to remove residual noyse that survived to median filter

    rotation_corrected = pp.rotation_correction(binarized_images)

    dilated = pp.apply_dilation(rotation_corrected)

    """comparision = np.hstack((cv.bitwise_not(denoysed_images[0]), cv.bitwise_not(binarized_images[0])))

    cv.imshow('Comparision: Denoysed vs. dilated', comparision)
    cv.waitKey(0)"""

    pp.write_images(dilated, image_list)