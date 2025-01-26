import base64
from io import BytesIO
from PIL import Image

def compress_base64_image(base64_str, max_size_kb=1024):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    buffer = BytesIO()
    quality = 85

    while True:
        image.save(buffer, format="PNG", optimize=True, quality=quality)
        buffer_size_kb = buffer.tell() / 1024

        if buffer_size_kb <= max_size_kb or quality <= 10:  # Avoid very low quality
            break

        quality -= 5
        buffer.seek(0)
        buffer.truncate(0)

    compressed_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return compressed_base64