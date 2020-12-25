# Import required Image library
import glob
from PIL import Image
from tempfile import mkstemp
import shutil

"""
BAD_PATH_NAME = "./Data/CK+YS+DH+JE/Bad/"
BAD_UPDATE_PATH_NAME = "./Data/Fixed_CK_YS_DH_JE/Bad/"
bad_images = glob.glob(f"{BAD_PATH_NAME}/*.jpg")
bad_updated_images = glob.glob(f"{BAD_UPDATE_PATH_NAME}/*.jpg")

GOOD_PATH_NAME = "./Data/CK+YS+DH+JE/Good/"
GOOD_UPDATE_PATH_NAME = "./Data/Fixed_CK_YS_DH_JE/Good/"
good_images = glob.glob(f"{GOOD_PATH_NAME}/*.jpg")
good_updated_images = glob.glob(f"{GOOD_UPDATE_PATH_NAME}/*.jpg")

"""
BAD_CHAN_PATH_NAME = "./Data/YS+JE/YS/Bad/"
BAD_CHAN_UPDATE_PATH_NAME = "./Data/Fixed_YS+JE/YS/Bad/"
bad_chan_images = glob.glob(f"{BAD_CHAN_PATH_NAME}/*.jpg")
bad_chan_updated_images = glob.glob(f"{BAD_CHAN_UPDATE_PATH_NAME}/*.jpg")


GOOD_CHAN_PATH_NAME = "./Data/YS+JE/YS/Good/"
GOOD_CHAN_UPDATE_PATH_NAME = "./Data/Fixed_YS+JE/YS/Good/"
Good_chan_images = glob.glob(f"{GOOD_CHAN_PATH_NAME}/*.jpg")
Good_chan_updated_images = glob.glob(f"{GOOD_CHAN_UPDATE_PATH_NAME}/*.jpg")

for img in bad_chan_images:
    # Create an Image Object from an Image
    im = Image.open(img)
    file_name = img.split("/")[-1]

    # Display actual image
    # im.show()

    img_size = im.size
    print(img_size)

    if img_size[0] > 320 and img_size[1] > 240:

        # Make the new image half the width and half the height of the original image
        resized_im = im.resize((round(im.size[0] * 0.5), round(im.size[1] * 0.5)))

        # Display the resized imaged
        # resized_im.show()

        # Save the cropped image
        resized_im.save(BAD_CHAN_UPDATE_PATH_NAME + file_name)
    else:
        shutil.copyfile(img, BAD_CHAN_UPDATE_PATH_NAME + file_name)


for img in Good_chan_images:
    # Create an Image Object from an Image
    im = Image.open(img)
    file_name = img.split("/")[-1]

    # Display actual image
    # im.show()

    img_size = im.size
    print(img_size)

    if img_size[0] > 320 and img_size[1] > 240:

        # Make the new image half the width and half the height of the original image
        resized_im = im.resize((round(im.size[0] * 0.5), round(im.size[1] * 0.5)))

        # Display the resized imaged
        # resized_im.show()

        # Save the cropped image
        resized_im.save(GOOD_CHAN_UPDATE_PATH_NAME + file_name)
    else:
        shutil.copyfile(img, GOOD_CHAN_UPDATE_PATH_NAME + file_name)

"""
for img in bad_updated_images:
    # Create an Image Object from an Image
    im = Image.open(img)
    file_name = img.split("/")[-1]

    # Display actual image
    # im.show()

    img_size = im.size
    print(img_size)

    if img_size[0] > 320 and img_size[1] > 240:
        print(f"{file_name} not resized")


for img in good_updated_images:
    # Create an Image Object from an Image
    im = Image.open(img)
    file_name = img.split("/")[-1]

    # Display actual image
    # im.show()

    img_size = im.size
    print(img_size)

    if img_size[0] > 320 and img_size[1] > 240:
        print(f"{file_name} not resized")
"""
