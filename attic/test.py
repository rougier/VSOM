import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


fig = plt.figure(figsize=(12,5))
gs = GridSpec(1, 10, figure=fig)

axs = []
axs.append (fig.add_subplot(gs[:,:3]))
axs.append (fig.add_subplot(gs[:,3:-3]))
axs.append (fig.add_subplot(gs[:,-3:]))


#fig, axs = plt.subplots(1, 3, constrained_layout=True)

img = imageio.imread('mucha.png')
data = np.zeros((256,256))

axs[0].set_aspect(img.shape[1]/img.shape[0])
axs[0].imshow(img)
axs[0].set_xticks([]), axs[0].set_yticks([])

axs[1].set_aspect(1)
axs[1].set_xticks([]), axs[1].set_yticks([])
axs[1].imshow(data)
    
axs[2].set_aspect(img.shape[1]/img.shape[0])
axs[2].imshow(img)
axs[2].set_xticks([]), axs[2].set_yticks([])

plt.tight_layout()
plt.show()
