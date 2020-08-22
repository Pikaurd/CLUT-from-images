import numpy as np
from PIL import Image
from typing import List


def foo(filter_path: str) -> None:
    lookup_filter = Image.open(filter_path)
    filter_bitmap = np.array(lookup_filter)

    sample_image = Image.open('../sample.png')
    # sample_image = Image.open('4x4.png')
    sample_bitmap = np.array(sample_image)
    
    target_bitmap = np.zeros(sample_bitmap.shape)

    is_debug_mode = False
    for i in range(target_bitmap.shape[0]):
        for j in range(target_bitmap.shape[1]):
            x = color_lookup(filter_bitmap, sample_bitmap[i][j], is_debug = is_debug_mode)
            target_bitmap[i, j] = x


    target_image = Image.fromarray(target_bitmap.astype('uint8')).convert('RGBA')
    target_image.save('a.png')

    # for i in range(63):
    # x = color_lookup(filter_bitmap, np.array([0, 0, 1, 0]), is_debug = is_debug_mode)

    print('x')


def apply_filter(filter_path: str, source_path: str, target_path: str) -> None:
    lookup_filter = Image.open(filter_path)
    filter_bitmap = np.array(lookup_filter)

    sample_image = Image.open(source_path)
    sample_bitmap = np.array(sample_image)
    
    target_bitmap = np.zeros(sample_bitmap.shape)

    is_debug_mode = False
    for i in range(target_bitmap.shape[0]):
        for j in range(target_bitmap.shape[1]):
            x = color_lookup(filter_bitmap, sample_bitmap[i][j], is_debug = is_debug_mode)
            target_bitmap[i, j] = x


    target_image = Image.fromarray(target_bitmap.astype('uint8')).convert('RGBA')
    target_image.save(target_path) 

    
def color_lookup(lookup_table: np.array, color: np.array, is_debug: bool = False) -> np.array:
    origin_r = color[0]
    origin_g = color[1]
    origin_b = color[2]
    origin_a = color[3]

    lookup_block_x = origin_b // 4 // 8
    lookup_block_y = origin_b // 4 % 8

    lookup_coordinates_offset_x = lookup_block_x * 64
    lookup_coordinates_offset_y = lookup_block_y * 64

    lookup_coordinates_x = lookup_coordinates_offset_x + origin_r // 4
    lookup_coordinates_y = lookup_coordinates_offset_y + origin_g // 4

    target_color = lookup_table[lookup_coordinates_x][lookup_coordinates_y]
    target_color_with_alpha_channel = np.zeros((1,4))[0]
    target_color_with_alpha_channel[0] = target_color[1]  # 为啥反了呢…………
    target_color_with_alpha_channel[1] = target_color[0]
    target_color_with_alpha_channel[2] = target_color[2]
    target_color_with_alpha_channel[3] = origin_a

    if is_debug:
        print('in {} \t out: {} \t block position: ({}, {})'.format(color, target_color_with_alpha_channel, lookup_block_x, lookup_block_y))

    return target_color_with_alpha_channel


if __name__ == '__main__':
    # filter_path = '../filter_resource/Identity/lookup.png'
    # foo(filter_path)


    filter_resource_pattern = 'filter_toast/resource/filter_resource/*'
    import glob
    import os.path

    sample = 'filter_toast/resource/sample.png'
    for filter_path in glob.glob(filter_resource_pattern):
        print('processing %s' % filter_path)
        filter_name = os.path.split(filter_path)[-1]
        filter_resource = os.path.join(filter_path, 'lookup.png')
        apply_filter(filter_resource, sample, 'filter_toast/output/' + filter_name + '.png')

