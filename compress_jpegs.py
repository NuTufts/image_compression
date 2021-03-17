from PIL import Image
import os


input_directory = '/home/jhwang/image_compression/results/originals/originals_png/'
output_directory = '/home/jhwang/image_compression/results/originals/originals_jpeg_compression/'
for i, file in enumerate(os.listdir(input_directory)):
    if file.endswith(".png"):
        print(f"Compressed: {i}", end='\r')
        img = Image.open(os.path.join(input_directory, file))
        img.save(output_directory+file[:-3]+'jpeg', optimize=True, quality=50)
print()
print('Fin')        
