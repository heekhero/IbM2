import os
import shutil

# define path
cub_data_dir = '/path/to/cub_200_2011/'


images_dir = os.path.join(cub_data_dir, 'images')
splits_file = os.path.join(cub_data_dir, 'train_test_split.txt')
image_list_file = os.path.join(cub_data_dir, 'images.txt')

# create train and test dir
train_dir = os.path.join(cub_data_dir, 'train')
test_dir = os.path.join(cub_data_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# read train_test_split.txt
with open(splits_file, 'r') as f:
    splits = [line.strip().split() for line in f.readlines()]


with open(image_list_file, 'r') as f:
    image_list = [line.strip().split() for line in f.readlines()]


image_dict = {image_id: image_path for image_id, image_path in image_list}


# copy files
for image_id, is_train in splits:
    image_path = image_dict[image_id]
    src_path = os.path.join(images_dir, image_path)

    if is_train == '1':
        dst_path = os.path.join(train_dir, image_path)
    else:
        dst_path = os.path.join(test_dir, image_path)

    dst_folder = os.path.dirname(dst_path)
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src_path, dst_path)

print('Image split completed.')
