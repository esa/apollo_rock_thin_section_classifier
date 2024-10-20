import matplotlib.pyplot as plt


def plot_single_image(image, title='Original image', save_name='Original image', save=True):
    """Plots a single image.

    Args:
        image: A numpy array representing the image with shape (height, width, channels)
    """

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(image)
    ax.axis('off')
    plt.gca().set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_name}.png', dpi=500)
    plt.show()
    
def plot_augmented_images(images):
  """Plots the provided images in a grid.

  Args:
      images: A numpy array of images with shape (num_images, height, width, channels).
  """

  num_images = images.shape[0]
  num_cols = (num_images + 1) // 2  # Calculate number of rows for the grid
  fig, axes = plt.subplots(2, num_cols, figsize=(2*num_cols, 4))

  for i, ax in enumerate(axes.flat):
      if i < num_images:
          ax.imshow(images[i])
          ax.set_title(f"Augmentation {i+1}")
          ax.set_xticks([])  # Remove x-axis tick labels
          ax.set_yticks([])  # Remove y-axis tick labels          
      else:
          ax.axis('off')  # Hide empty subplot if num_images is odd

  plt.tight_layout()
  plt.savefig('Augmented images.png', dpi=500)
  
  plt.show()
