import argparse
import omegaconf

import matplotlib.pyplot as plt

import io
import base64


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_to_gen_html(sample, title): 
    # create a figure and plot sample on it 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sample, cmap='gray')
    ax.axis('off')

    # Add title 
    ax.set_title(title, fontsize=15)
    
    # Convert the Matplotlib figure to a PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_png = buf.getvalue()
    
    # Encode the PNG image to base64 string
    img_str = base64.b64encode(img_png).decode('utf-8')
    
    buf.close()
    plt.close(fig)  # Close the figure to free memory

    return img_str 