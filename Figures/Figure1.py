import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import transforms
import h5py

SetLabelSize = 24
SetAnnoSize = 18
SetLegendSize = 16
SetMarkerSize = 8

# Create figure and axis
fig, ax = plt.subplots(figsize=(7.0, 7.0))

# Define radii
R = 1
rho = 2.5

# Draw circles with distinct colors and transparency
circle2 = plt.Circle((0, 0), 10, color='palegoldenrod', fill=True, alpha=0.8)
circle1 = plt.Circle((0, 0), rho, color='#ADD8E6', fill=True, alpha=0.7)  # Light blue
circle0 = plt.Circle((0, 0), R, color='#FFA500', fill=True, alpha=0.7)  # Orange

# Add circles to axis in the correct order (outer to inner)
ax.add_artist(circle2)
ax.add_artist(circle1)
ax.add_artist(circle0)

file = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\TarFix_10P600_2.h5'
with h5py.File(file, 'r') as hdf:
    groupname = f"case_{6}"
    group = hdf.get(groupname)
    loc_x = group.get('tx')  # get the x-coordinate
    loc_y = group.get('ty')  # get the y-coordinate
    loc_x = loc_x[:]*10-5
    loc_y = loc_y[:]*10-5
    ax.scatter(loc_x, loc_y, c='forestgreen', marker = '^', s=45,  alpha = 0.7, edgecolors='none', label = 'Targets')

# Add radii labels
ax.annotate('', xy=(-R/np.sqrt(2), R/np.sqrt(2)), xytext=(0, 0), color='black', fontsize=SetLabelSize, arrowprops=dict(facecolor='black',
        shrink=0,width=2,headwidth=9,headlength=12))
ax.annotate('', xy=(rho/np.sqrt(2), -rho/np.sqrt(2)), xytext=(0, 0), color='black', fontsize=SetLabelSize, arrowprops=dict(facecolor='black',
        shrink=0,width=2,headwidth=9,headlength=12))
ax.annotate(r'$R$', xy=(-R/np.sqrt(2), R/np.sqrt(2)), xytext=(-R/np.sqrt(2)-0.2, R/np.sqrt(2)+0.2), color='black', fontsize=SetLabelSize)
ax.annotate(r'$\rho$', xy=(rho/np.sqrt(2), -rho/np.sqrt(2)), xytext=(rho/np.sqrt(2)+0.2, -rho/np.sqrt(2)-0.2), color='black', fontsize=SetLabelSize)

# Function to add an image (sticker) at specific coordinates
def add_sticker(ax1, x, y, image_path, zoom=0.1, rotation=0):
    # Load the image
    img = plt.imread(image_path)
    # Create an OffsetImage with specified zoom (size)
    imagebox = OffsetImage(img, zoom=zoom)
    # Apply rotation: Create a rotation transform
    rotation_transform = transforms.Affine2D().rotate_deg(rotation)
    # Combine with the axis transform
    imagebox.set_transform(rotation_transform + ax.transData)
    # Create an AnnotationBbox to place the image at (x, y)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    # Add the annotation to the axis
    ax1.add_artist(ab)

sticker_path = 'D:\\OneDrive - University of Pittsburgh\\CollectiveForagingData\\6000T\\ant.png'
add_sticker(ax,0, 0, sticker_path, rotation=0)
add_sticker(ax,-3, 3, sticker_path, rotation=60)
add_sticker(ax,-3, -3, sticker_path, rotation=60)
add_sticker(ax,3, -3, sticker_path, rotation=60)
add_sticker(ax,3, 3, sticker_path, rotation=60)
add_sticker(ax,0, -2, sticker_path, rotation=60)

poss_tx = loc_x[(loc_x < R) & (loc_y > -R) & (loc_x > 0) & (loc_y < 0)]
poss_ty = loc_y[(loc_x < R) & (loc_y > -R) & (loc_x > 0) & (loc_y < 0)]
ax.scatter(poss_tx[0], poss_ty[0], facecolors='none', edgecolors='tomato', marker='o', s=100, alpha=1, label='Detected Target')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
## plot lines for exploration
ax.plot([-0.5,-2, 0], [-2.9, -1.8, 0], color='m',  marker='o', linestyle='-', markersize=SetMarkerSize, label=r'$\mu=1.1$', lw=2)
ax.annotate('', xy=(-1.25, -2.35), xytext=(-0.5,-2.9), color='black', fontsize=SetLabelSize,
            arrowprops=dict(arrowstyle='-|>', facecolor='m', linestyle='None'))
ax.annotate('', xy=(-1, -0.9), xytext=(-2,-1.8), color='black', fontsize=SetLabelSize,
            arrowprops=dict(arrowstyle='-|>', facecolor='m', linestyle='None'))
ax.plot([-0.5,-2, -8], [-2.9, -4, -6], color='m', marker='o', linestyle='-', markersize=SetMarkerSize, lw=2)
ax.annotate('', xy=(-1.25,-3.45), xytext=(-2,-4), color='black', fontsize=SetLabelSize,
            arrowprops=dict(arrowstyle='-|>', facecolor='m', linestyle='None'))
ax.annotate('', xy=(-3.5, -4.5), xytext=(-5,-5), color='black', fontsize=SetLabelSize,
            arrowprops=dict(arrowstyle='-|>', facecolor='m', linestyle='None'))
# ax.annotate(r'$\mu=1.1$', xy=(-2, -1.8), xytext=(-3.2, -2.4), color='black', fontsize=SetAnnoSize)
## plot lines for exploitation
# ax.plot([0.6, 0.9, 0.3, 0], [0.4, 1, 0.7, 0], color='c', marker='o', linestyle='-', markersize=SetMarkerSize, label = r'$\mu=3$', lw=2)
# ax.annotate('', xy=(0.2, 0.46), xytext=(0,0), color='black', fontsize=SetLabelSize,
#             arrowprops=dict(arrowstyle='-|>', facecolor='c', linestyle='None'))
# ax.annotate('', xy=(0.6, 0.85), xytext=(0.3, 0.7), color='black', fontsize=SetLabelSize,
#             arrowprops=dict(arrowstyle='-|>', facecolor='c', linestyle='None'))
# ax.annotate('', xy=(0.75, 0.7), xytext=(0.9, 1), color='black', fontsize=SetLabelSize,
#             arrowprops=dict(arrowstyle='-|>', facecolor='c', linestyle='None'))
# ax.plot([0.6, 0.9, 0.5], [0.4, 0.2, -0.1], color='c', marker='o', linestyle='-', markersize=SetMarkerSize, lw=2)
# ax.plot([0.5, 0.3, 0.9, 1.3], [-0.1, 0.3, 0.5, 0.1], color='c', marker='o', linestyle='-', markersize=SetMarkerSize, lw=2)

# Define the 11 points forming 10 segments
x = [0, 0.3, 0.9, 0.6, 0.9, 0.5, 0.3, 0.9, 1.3]
y = [0, 0.7, 1, 0.4, 0.2, -0.1, 0.3, 0.5, 0.1]

# Plot the polyline
ax.plot(x, y, color='c', marker='o', linestyle='-', markersize=SetMarkerSize,
        label=r'$\mu=3$', lw=2)

# Add an arrow for each of the 10 segments
for i in range(len(x) - 1):  # Loop over segments (0 to 9)
    x_start, y_start = x[i], y[i]  # Start of segment
    x_end, y_end = x[i + 1], y[i + 1]  # End of segment

    # Calculate arrow head position (2/3 along the segment)
    t = 1/2
    x_head = x_start + t * (x_end - x_start)
    y_head = y_start + t * (y_end - y_start)

    # Calculate arrow tail position (1/3 along the segment for a short arrow)
    t_tail = 3 / 4
    x_tail = x_start + t_tail * (x_end - x_start)
    y_tail = y_start + t_tail * (y_end - y_start)

    # Add the arrow
    ax.annotate('', xy=(x_tail, y_tail), xytext=(x_head, y_head), color='black', fontsize=SetLabelSize,
             arrowprops=dict(arrowstyle='-|>', facecolor='c', linestyle='None'))

## plot lines for targeted walk
# find the detected target
ax.plot([0, poss_tx[0]], [-2, poss_ty[0]], color='tomato', linestyle='-', label = 'Targeted Walk', lw=2)
ax.annotate('', xy=(poss_tx[0]/2, (poss_ty[0]-2)/2), xytext=(0,-2), color='black', fontsize=SetLabelSize,
            arrowprops=dict(arrowstyle='-|>', facecolor='tomato', linestyle='None'))

# Set aspect ratio to equal
ax.set_aspect('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
ax.legend(loc='upper right', ncol=1, fontsize=SetLegendSize, frameon=True)

# Remove axes
ax.axis('off')
# Adjust layout to remove white margins
plt.tight_layout(pad=0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# Save the figure in a high-quality PDF format (vector graphics)
plt.savefig('Figure1.diagram.pdf', format='pdf', dpi=600)
plt.savefig('Figure1.diagram.png', format='png', dpi=600)
# Show plot
plt.show()