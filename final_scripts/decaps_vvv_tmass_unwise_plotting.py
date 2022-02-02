import matplotlib
matplotlib.use('pdf')
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def make_los_fig(dispsurf,los_samples,maxdistplot,maxredplot,Av_upper=7,ebin_num=700,dbin_num=120,y_max=7):

    xpts = np.linspace(4, 19, dbin_num)
    ypts = np.linspace(0, Av_upper, ebin_num)
    x_range = [4, 19]
    y_range = [0, Av_upper]

    #Set up figure
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, adjustable='box', aspect='equal')
    ax.set_xlabel(
        r'$\mu$ (mag)',
        fontsize=25,
        labelpad=-20,
        position=(
            0.00,
            0.3),
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=1.,
            pad=0.2))
    ax.set_ylabel(r'E(B-V) (mag)', fontsize=25, labelpad=5)

    #This will be inputed into imshow to convert from pixel scale to correct distance-extinction scale
    if xpts is not None and ypts is not None:
        dx = np.median(xpts[1:] - xpts[:-1])
        dy = np.median(ypts[1:] - ypts[:-1])
        extent = [xpts[0] - dx / 2., xpts[-1] + dx /
                  2., ypts[0] - dy / 2., ypts[-1] + dy / 2.]

    #This shows the stacked stellar posteriors for the star in the background with a grayscale
    ax.imshow(
        np.sqrt(dispsurf),
        origin='lower',
        aspect='auto',
        cmap='binary',
        interpolation='nearest',
        extent=extent)
    ax.autoscale(False)

    #set the limits
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_max)

    #mark the most likely distance and extinction for each star with a red cross
    ax.scatter(
        maxdistplot,
        maxredplot,
        c='r',
        marker='+',
        s=25,
        alpha=0.75,
        zorder=1)
        
        
    plot_x=np.linspace(4, 19, 121)[:-1]+0.0625

    for i in range(2,102):
        plot_y_samp=los_samples[0,i,2:]*0.01
        plt.plot(plot_x,plot_y_samp,c='blue',lw=0.5,alpha=0.25)
        
    best_sample =  los_samples[0,1,2:]
    plot_y_best=best_sample*0.01
    plt.plot(plot_x,plot_y_best,c='black',lw=3)
    plt.plot(plot_x,plot_y_best,c='yellow',lw=2)

    #set tick parameters
    ax.xaxis.set_tick_params(labelsize=18, zorder=2)
    ax.yaxis.set_tick_params(labelsize=18, zorder=2)
    

def compute_plot_objects(surfs,ebin_num=700.,dbin_num=120.,normcol=False, Av_upper=7.):
    
    surf_list=[surfs[i,:,:] for i in range(surfs.shape[0])]
    maxlike=[np.where(surf_list[i] == surf_list[i].max()) for i in range(len(surf_list))]
    maxred=[maxlike[i][0][0] for i in range(len(maxlike))]
    maxdist=[maxlike[i][1][0] for i in range(len(maxlike))]
    maxredplot=np.array(maxred)*(Av_upper/ebin_num)
    maxdistplot=4+np.array(maxdist)*(15.0/dbin_num)
    
    dispsurf=surfs
    dispsurf[~np.isfinite(dispsurf)] = 0
    norm = np.sum(np.sum(dispsurf, axis=1), axis=1)+1.e-40
    totsurf = np.sum(dispsurf/norm.reshape(-1,1,1), axis=0)

    if normcol==True:
        #normalize so every distance column has same amount of ink
        colmin = np.min(totsurf, axis=0)
        dispsurf = totsurf-colmin.reshape(1,-1)
        dispsurf = dispsurf/np.sum(dispsurf, axis=0).reshape(1,-1)
    
    else:
        dispsurf=totsurf.copy()

    return dispsurf,maxdistplot,maxredplot
    
filename_index = sys.argv[1]

f_in=h5py.File('/n/fink2/czucker/Plane_Final/perstar/G314/G314.{:05d}.h5'.format(int(filename_index)),'r')
f_out=h5py.File('/n/fink2/czucker/Plane_Final/output/it0/G314/G314.{:05d}.h5'.format(int(filename_index)),'r')

for pix in f_in['stellar_pdfs'].keys():
   
    pdfs=f_in['stellar_pdfs/%s/stellar_pdfs'%(pix)][:]
    l = np.round(f_in['stellar_pdfs/%s'%(pix)].attrs['l'],2)
    b = np.round(f_in['stellar_pdfs/%s'%(pix)].attrs['b'],2)
        
    los_samples=f_out['%s/discrete-los'%(pix)][:]

    dispsurf, maxdistplot, maxredplot = compute_plot_objects(pdfs,normcol=False, ebin_num=700., dbin_num=120., Av_upper=7)

    make_los_fig(dispsurf,los_samples,maxdistplot,maxredplot)
    
    plt.title('File {}: {} (l, b = {}, {})'.format(filename_index, pix,l,b))

    plt.savefig("/n/fink2/czucker/Plane_Final/plots/G314/{}_los.png".format(pix.replace(' ','_')),bbox_inches='tight',dpi=70)

    plt.close('all')
