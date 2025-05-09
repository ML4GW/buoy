import base64
from pathlib import Path


def html_header(label: str) -> str:
    """
    Generate the HTML header with a title.

    Args:
        label (str): Title for the HTML page.

    Returns:
        str: HTML header string.
    """
    html_header = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{label}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                margin: 0;
                padding: 20px;
            }}
            .gallery {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }}
            .item {{
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
                box-sizing: border-box;
            }}
            img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .caption {{
                margin-top: 8px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h1>{label}</h1>
        <div class="gallery">
    """
    return html_header


html_footer = """
    </div>
</body>
</html>
"""


def embed_image(image_path: Path, caption: str) -> str:
    """
    Embed an image in HTML using base64 encoding.

    Args:
        image_path (str): Path to the image file.
        caption (str): Caption for the image.

    Returns:
        str: html string with the embedded image.
    """
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f'''
        <div class="item">
            <img src="data:image/png;base64,{encoded_string}" alt="{caption}">
            <div class="caption">{caption}</div>
        </div>
    '''


def generate_html(
    plotdir: Path, output_file: Path, label: str = "Event Summary"
) -> None:
    """
    Generate an HTML file with embedded images from a directory.

    Args:
        plotdir (Path): Directory containing the images.
        output_file (Path): Output HTML file path.
        label (str): Title for the HTML page.
    """
    html_content = html_header(label)
    for image_path in plotdir.glob("*.png"):
        caption = image_path.stem
        caption = caption.replace("_", " ")
        caption = (
            caption[0].upper() + caption[1:]
        )  # Capitalize the first letter
        html_content += embed_image(image_path, caption)
    html_content += html_footer

    with open(output_file, "w") as f:
        f.write(html_content)
