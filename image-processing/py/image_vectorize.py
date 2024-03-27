from PIL import Image

def get_vector_from_image(path):
    image = Image.open(path)
    width, height = image.size

    gray_vector = []

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            intensity = 255 - int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            gray_vector.append(intensity)

    return gray_vector


