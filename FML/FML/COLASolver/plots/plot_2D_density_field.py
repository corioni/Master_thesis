import numpy as np
import plotting_library as PL
from pylab import *
from matplotlib.colors import LogNorm
import matplotlib

# =============================================
# Set plotting defaults
# =============================================
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12 })
matplotlib.rcParams['text.usetex'] = True
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

#snapshot name
snapshot = '../output/particles/snapshot_Sim_Test_z0.000/gadget_z0.000.0' 

# density field parameters
x_min, x_max = 0, 8
y_min, y_max = 0, 512
z_min, z_max = 0, 512 
grid         = 2048 
ptypes       = [1]   # 0-Gas, 1-CDM, 2-NU, 4-Stars; can deal with several species
plane        = 'YZ'  #'XY','YZ' or 'XZ'
MAS          = 'CIC' #'NGP', 'CIC', 'TSC', 'PCS'
save_df      = False  #whether save the density field into a file

# image parameters
fout            = 'Image_slice.pdf'
min_overdensity = None      #minimum overdensity to plot
max_overdensity = None   #maximum overdensity to plot
scale           = 'log' #'linear' or 'log'
cmap            = 'viridis'


# compute 2D overdensity field
dx, x, dy, y, overdensity = PL.density_field_2D(snapshot, x_min/1e3, x_max/1e3, y_min/1e3, y_max/1e3,
                                                z_min/1e3, z_max/1e3, grid, ptypes, plane, MAS, save_df)


# plot density field
print('\nCreating the figure...')
fig = figure()    #create the figure
ax1 = fig.add_subplot(111)

ax1.set_xlim([x, x+dx])  #set the range for the x-axis
ax1.set_ylim([y, y+dy])  #set the range for the y-axis

ax1.set_xlabel(r'$h^{-1}{\rm Gpc}$')  #x-axis label
ax1.set_ylabel(r'$h^{-1}{\rm Gpc}$')  #y-axis label

if min_overdensity==None:  min_overdensity = np.min(overdensity)+1e-1
if max_overdensity==None:  max_overdensity = np.max(overdensity)

overdensity[np.where(overdensity<min_overdensity)] = min_overdensity

if scale=='linear':
      cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                       extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                       vmin=min_overdensity,vmax=max_overdensity)
else:
      cax = ax1.imshow(overdensity,cmap=get_cmap(cmap),origin='lower',
                       extent=[x, x+dx, y, y+dy], interpolation='bicubic',
                       norm = LogNorm(vmin=min_overdensity+1,vmax=max_overdensity-100))

cbar = fig.colorbar(cax)
cbar.remove()
savefig(fout, bbox_inches='tight')
close(fig)
