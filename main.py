import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
from PIL import Image as PILImage
from io import BytesIO
import requests

if 'used_colors' not in st.session_state:
    st.session_state.used_colors = []
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'color_updated' not in st.session_state:
    st.session_state.color_updated = False
if 'params' not in st.session_state:
    st.session_state.params = {}


def create_color_label(color, size=100, text_height=30):
    square = create_color_square(color, size=size)
    label = Image.new('RGB', (size, size + text_height), (255, 255, 255))
    label.paste(square, (0, 0))
    draw = ImageDraw.Draw(label)
    hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    text_width = font.getlength(hex_color)
    x = (size - text_width) // 2
    draw.text((x, size + 5), hex_color, fill=(0, 0, 0), font=font)
    return label


def resize_image_with_padding(image, target_width, target_height, bg_color=(0, 0, 0)):
    h, w, _ = image.shape
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Convert hex color to BGR
    if isinstance(bg_color, str):
        bg_color = tuple(int(bg_color.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))

    padded_image = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    top_offset = (target_height - new_h) // 2
    left_offset = (target_width - new_w) // 2
    padded_image[top_offset:top_offset + new_h, left_offset:left_offset + new_w] = resized_image
    return padded_image


def limit_colors(image, num_colors=7, enhance_brightness=True, enhance_saturation=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    labels = kmeans.labels_

    # Enhance colors directly in cluster centers
    hsv_centers = cv2.cvtColor(np.array([kmeans.cluster_centers_], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0]
    if enhance_saturation:
        hsv_centers[:, 1] = np.clip(hsv_centers[:, 1] * 1.5, 0, 255)
    if enhance_brightness:
        hsv_centers[:, 2] = np.clip(hsv_centers[:, 2] * 1.2, 0, 255)
    enhanced_centers = cv2.cvtColor(np.array([hsv_centers], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0]

    quantized_image = enhanced_centers[labels].reshape(image.shape).astype(np.uint8)
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR)

    return quantized_image, enhanced_centers, labels


def enlarge_clusters(image, block_size=10):
    h, w = image.shape[:2]
    enlarged = np.zeros((h * block_size, w * block_size, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            color = image[y, x]
            enlarged[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = color

    return enlarged


def draw_grid(image, line_width=2, line_spacing=8):
    height, width, _ = image.shape

    # Create a copy of the image to draw the grid on
    image_with_grid = image.copy()

    # Draw vertical lines
    for x in range(0, width, line_spacing + line_width):
        cv2.line(image_with_grid, (x, 0), (x, height), (0, 0, 0), line_width)

    # Draw horizontal lines
    for y in range(0, height, line_spacing + line_width):
        cv2.line(image_with_grid, (0, y), (width, y), (0, 0, 0), line_width)

    return image_with_grid


def create_color_square(color, size=50, frame_thickness=2):
    """
    Create a color square with a black frame.
    :param color: RGB color tuple (e.g., (255, 0, 0))
    :param size: Size of the square in pixels
    :param frame_thickness: Thickness of the black frame
    :return: PIL Image object
    """
    # Create a blank image with the specified size
    square = PILImage.new('RGB', (size, size), color)

    # Draw a black frame around the square
    for i in range(frame_thickness):
        square.paste((0, 0, 0), (i, i, size - i, size - i))
        square.paste(color,
                     (i + frame_thickness, i + frame_thickness, size - i - frame_thickness, size - i - frame_thickness))

    return square


# Streamlit application
st.title("Neulontaohje kuvasta")

# Then modify your file uploader like this:
uploaded_file = st.file_uploader(
    "Valitse kuva omista kuvistasi...",
    type=["jpg", "png", "jpeg", "bmp", "gif"],
    help="Raahaa ja pudota kuva tähän tai valitse tiedosto",
    key="uploaded_file"
)

def set_example_text():
    st.session_state.url = "https://i.media.fi/incoming/2013/09/11/sininen.jpg/alternates/FREE_960/sininen.jpg"

# Initialize session state for the text input if it doesn't already exist
if 'url' not in st.session_state:
    st.session_state['url'] = ""

# Text input with a placeholder and key for session state management
url = st.text_input(
    "...Tai anna kuvan URL",
    placeholder="",
    key="url"
)

# Display example text with a button
st.button("...Tai käytä alla olevaa esimerkkiä painamalla tästä", on_click=set_example_text)
st.write("https://i.media.fi/incoming/2013/09/11/sininen.jpg/alternates/FREE_960/sininen.jpg")

if st.session_state.uploaded_file or st.session_state.url:

    if not uploaded_file:
        response = requests.get(url)
        uploaded_file = BytesIO(response.content)

    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply bilateral filter with increased parameters to reduce noise while preserving edges
    filtered_image = cv2.bilateralFilter(image, d=15, sigmaColor=150, sigmaSpace=150)

    # Apply Gaussian blur with a larger kernel size
    image = cv2.GaussianBlur(filtered_image, (15, 15), 0)

    st.image(image)

    # Input parameters
    height_cm = st.number_input("Haluttu korkeus (cm):", min_value=1, max_value=100, value=50)
    width_cm = st.number_input("Haluttu leveys (cm):", min_value=1, max_value=100, value=50)
    count_colors = st.number_input("Kuinka monella värillä tehdään?", min_value=1, max_value=20, value=7)
    loops_per_10 = st.number_input("Silmukoiden määrä 10 senttimetrissä:", min_value=1, max_value=100, value=22)
    layers_per_10 = st.number_input("Kerrosten määrä 10 senttimetrissä:", min_value=1, max_value=100, value=30)
    background_color = st.color_picker("Valitse taustaväri tyhjälle alueelle:", "#FFFFFF")

    grid_width = int((width_cm / 10) * loops_per_10)
    grid_height = int((height_cm / 10) * layers_per_10)

    current_params = {
        'height_cm': height_cm,
        'width_cm': width_cm,
        'count_colors': count_colors,
        'loops_per_10': loops_per_10,
        'layers_per_10': layers_per_10,
        'background_color': background_color,
        'grid_width': grid_width,
        'grid_height': grid_height
    }

    block_size = 16
    line_spacing = block_size - 2

    # Check if any parameters have changed
    current_file = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else None
    params_changed = (
            (st.session_state.params != current_params) or
            (st.session_state.uploaded_file != current_file)
    )

    # Store current parameters
    st.session_state.params = current_params

    # Process image only if parameters changed or new file uploaded
    if params_changed:
        image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        padded_image = resize_image_with_padding(image, grid_width, grid_height, background_color)
        quantized_image, used_colors, labels = limit_colors(padded_image, num_colors=count_colors)

        st.session_state.base_image = padded_image
        st.session_state.used_colors = used_colors.tolist()
        st.session_state.labels = labels
        try:
            st.session_state.uploaded_file = uploaded_file.getvalue()
        except:
            pass

        st.session_state.used_colors = used_colors.tolist()  # Store as list for easier modification
        st.session_state.color_updated = True

    st.write("Lankojen värit (muuta värivalitsimilla):")

    # Create rows of 5 color pickers
    cols_per_row = 5
    color_cols = st.columns(cols_per_row)

    for idx, color in enumerate(st.session_state.used_colors):
        with color_cols[idx % cols_per_row]:
            # Convert color to hex format
            hex_color = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"

            # Create color picker bound to the session state
            new_color = st.color_picker(
                label=" ",
                value=hex_color,
                key=f"color_{idx}",
                label_visibility="collapsed"
            )

            # Update color if changed
            if new_color != hex_color:
                r = int(new_color[1:3], 16)
                g = int(new_color[3:5], 16)
                b = int(new_color[5:7], 16)
                st.session_state.used_colors[idx] = [r, g, b]
                st.session_state.color_updated = True

    # Create new rows when needed
    if len(st.session_state.used_colors) > cols_per_row:
        for _ in range((len(st.session_state.used_colors) - 1) // cols_per_row):
            color_cols = st.columns(cols_per_row)

    # Regenerate image with current colors
    quantized_image = np.array(st.session_state.used_colors)[st.session_state.labels].reshape(
        st.session_state.base_image.shape)
    quantized_image = cv2.cvtColor(quantized_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    enlarged_image = enlarge_clusters(quantized_image, block_size=block_size)
    final_image = draw_grid(enlarged_image, line_width=2, line_spacing=line_spacing)
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    st.image(final_image_rgb, caption='Neulontaohje')

    # Generate color labels
    color_labels = []
    for color in st.session_state.used_colors:
        color_tuple = tuple(map(int, color))
        label = create_color_label(color_tuple)
        color_labels.append(label)

    # Create color strip
    square_size = 150
    text_height = 30
    spacing = 30
    top_space = 50
    total_width = len(color_labels) * square_size + (len(color_labels) - 1) * spacing
    total_height = square_size + text_height + top_space
    color_strip = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    x = 10
    for label in color_labels:
        color_strip.paste(label, (x, top_space))
        x += square_size + spacing

    # Convert main image to PIL
    main_image_pil = Image.fromarray(final_image_rgb)

    # Create combined image
    combined_width = max(color_strip.width, main_image_pil.width)
    combined_height = color_strip.height + main_image_pil.height
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    x_color = (combined_width - color_strip.width) // 2
    combined_image.paste(color_strip, (x_color, 0))
    x_main = (combined_width - main_image_pil.width) // 2
    combined_image.paste(main_image_pil, (x_main, color_strip.height))

    # Save to buffer
    buffer = BytesIO()
    combined_image.save(buffer, format="PNG", dpi=(300, 300))
    buffer.seek(0)

    # Add download button
    st.download_button(
        "Lataa tulostettava kuva",
        data=buffer,
        file_name="neulontaohje.png",
        mime="image/png"
    )
