# coding: utf-8
import imageio
import File.loop_file as loop_file

path = '/Users/lichen/Desktop/cifar_samples'
gif_path = '/Users/lichen/Desktop/cifar_samples.gif'

filenames = loop_file.list_dir(path, ['.png'])
print filenames

images = [imageio.imread(filename) for filename in filenames]
imageio.mimsave(gif_path, images, duration=1.0)