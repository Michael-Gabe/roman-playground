# Module imports
import numpy as np
import galsim
import sys
import yaml

with open(sys.argv[1]) as stream:
    pars = yaml.safe_load(stream)

print('#', pars)

# RNG Seed
rng_sim = np.random.default_rng(int(pars['seed']))
rng_recover = np.random.default_rng(int(pars['seed'])+2000000000)

# Weighting functions
def Omega4(x):
    flag = isinstance(x, np.ndarray)
    x_ = x
    if not flag:
        x_ = np.asarray([float(x)])
    t = np.clip(x,-1.,1.)
    y = -15./96.*t**7 +21./32.*t**5 -35./32.*t**3 +35./32.*t+.5
    return y

# Build galaxy image
def make_multi_gal_image(nimage, npix, psf_, gal, bkg, rng):
    """Generates npix x npix images of a galaxy with properties in gal.
    The psf is in psf_
    The background (float) is bkg
    Uses the random number generator rng
    
    Galaxy property gal should be a dictionary with 'r' (half light radius in pixels), 'g1' and 'g2' (shear), and
    'flux' (total flux)
    
    Output is of shape (nimage,npix,npix) (stack of images)
    the 0th image is noiseless
    """

    outcube = np.zeros((nimage,npix,npix))
    for j in range(nimage):
        image = galsim.ImageF(npix,npix,scale=1.)
    
        model_round = galsim.Exponential(half_light_radius=gal['r'], flux=gal['flux'])
        jshear = np.asarray([[1+gal['g1'], gal['g2']], [gal['g2'], 1-gal['g1']]])/np.sqrt(1-gal['g1']**2-gal['g2']**2)
        galimage = galsim.Convolve([galsim.Transformation(model_round, jac=jshear, offset=(0.,0.), flux_ratio=1), psf_])

        galimage.drawImage(image, add_to_image=True, method='no_pixel')
        I = image.array+bkg
        if j>0:
            I = rng.poisson(np.clip(I,0.,None))
        outcube[j,:,:] = I-bkg

        if j>0:
            outcube[j,:,:] = outcube[j,:,:] + gal['stripe']*rng.normal(size=npix)[:,None]
    return outcube

cut0 = float(pars['cut'][0]) # cosine-shaped cut-on for selecting galaxies
cut1 = float(pars['cut'][1])

if 'cutslope' in pars:
    sigmaref = float(pars['cutslope'][0])
    sigslope = float(pars['cutslope'][1])
else:
    sigmaref = 1.
    sigslope = 0.

# Re-measurement
def remeasure(image_):
    try:
        moms = galsim.Image(image_).FindAdaptiveMom()
        data_ = np.array([moms.moments_amp, moms.moments_centroid.x-xc-1, moms.moments_centroid.y-xc-1,
                         moms.moments_sigma, moms.observed_shape.g1, moms.observed_shape.g2])
        w = Omega4(-1+2*(data_[0]*(data_[3]/sigmaref)**sigslope-cut0)/(cut1-cut0))
        return(np.array([w,w*data_[4],w*data_[5]]))
    except:
        return(np.array([0.,0.,0.]))

Ngal = int(pars['Ngal'])
use_bkg = float(pars['bkg'])
xc = 25
nx = 2*xc+1

# Additional flags
add_stripe = float(pars['add_stripe']) # horizontal banding

# PSF
PSF = galsim.Gaussian(fwhm=3., flux=1.)

# Bias test
def bias_assess(F):
    nvar = 3
    galaxy_images = make_multi_gal_image(Ngal, nx, PSF, {'r': 4.0, 'g1': -.1, 'g2': 0.4, 'flux': F, 'stripe': add_stripe}, use_bkg, rng_sim)

    # Get the moments of the galaxies
    data = np.zeros((Ngal,6))
    nfail = 0
    is_failed = np.zeros(Ngal, dtype=bool)
    for j in range(Ngal):
        try:
            moms = galsim.Image(galaxy_images[j,:,:]).FindAdaptiveMom()
            data[j,:] = np.array([moms.moments_amp, moms.moments_centroid.x-xc-1, moms.moments_centroid.y-xc-1,
                         moms.moments_sigma, moms.observed_shape.g1, moms.observed_shape.g2])
            #if j==0: print('# ', data[0,3])
        except:
            data[j,:] = data[0,:]
            is_failed[j] = True
            nfail = nfail+1

    out = {'F_nn': data[0,0],
           'snr_det': np.sqrt(np.sum(galaxy_images[0,:,:]**2)/use_bkg),
           'snr': np.sqrt(np.sum(galaxy_images[0,:,:]**2/(use_bkg+galaxy_images[0,:,:])))
          }

    # Get noise realizations for each galaxy
    Nzeta = 10 # number of noise realizations per galaxy
    zeta = np.zeros((Nzeta,nx,nx))
    db0 = np.zeros((Ngal,nvar))
    db1 = np.zeros((Ngal,nvar))
    db2 = np.zeros((Ngal,nvar))

    if isinstance(pars['gauss'], str):
        useg = pars['gauss'].lower in ("yes", "true", "t", "1")
    else:
        useg = pars['gauss']
    K = Ngal
    for igal in range(K):
        im = galaxy_images[igal,:,:]
        db0[igal,:] = remeasure(im)
        ca = np.zeros(nvar)
        for j in range(Nzeta):
            if useg:
                zeta[j,:,:] = rng_recover.normal(scale=np.sqrt(use_bkg), size=(nx,nx))
            else:
                x = np.clip(im+use_bkg, 0., None)
                zeta[j,:,:] = x - rng_recover.gamma(x)
            if add_stripe>0:
                zeta[j,:,:] = zeta[j,:,:] + add_stripe*rng_recover.normal(size=nx)[:,None]
            ca = ca + remeasure(im + zeta[j,:,:])
        ca = ca/Nzeta - db0[igal,:]
        db1[igal,:] = db0[igal,:] - ca
        ca[:]=0.
        zeta = zeta/np.sqrt(2) # re-scale for what follows
        for j in range(0,Nzeta,2):
            ca = ca + remeasure(im +zeta[j,:,:]+zeta[j+1,:,:])\
                + remeasure(im +zeta[j,:,:]-zeta[j+1,:,:])\
                + remeasure(im -zeta[j,:,:]+zeta[j+1,:,:])\
                + remeasure(im -zeta[j,:,:]-zeta[j+1,:,:])\
               -2*remeasure(im +zeta[j,:,:])-2*remeasure(im -zeta[j,:,:])\
               -2*remeasure(im +zeta[j+1,:,:])-2*remeasure(im -zeta[j+1,:,:])\
               +4*db0[igal,:]
        ca = ca/(Nzeta//2)
        db2[igal,:] = db1[igal,:]+ca

    out['truth'] = db0[0,:]
    out['db0'] = np.mean(db0[1:K,:], axis=0)
    out['db1'] = np.mean(db1[1:K,:], axis=0)
    out['db2'] = np.mean(db2[1:K,:], axis=0)

    out['db0err'] = np.std(db0[1:K,:], axis=0)
    out['db1err'] = np.std(db1[1:K,:], axis=0)
    out['db2err'] = np.std(db2[1:K,:], axis=0)

    out['db0errm'] = np.std(db0[1:K,:], axis=0)/np.sqrt(K-2)
    out['db1errm'] = np.std(db1[1:K,:], axis=0)/np.sqrt(K-2)
    out['db2errm'] = np.std(db2[1:K,:], axis=0)/np.sqrt(K-2)

    # get covariances
    out['r0'] = np.nan_to_num(np.corrcoef(db0[1:K,:].T))
    out['r1'] = np.nan_to_num(np.corrcoef(db1[1:K,:].T))
    out['r2'] = np.nan_to_num(np.corrcoef(db2[1:K,:].T))
    return out

#print(bias_assess(4000.))
for k in range(67):
    F = 2000.+50.*k
    out = bias_assess(F)
    tr = out['truth'][0]
    s = '{:6.1f} {:5.2f}   {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:8.6f} {:8.6f} {:8.6f}'.format(F,
      out['snr'], tr, out['db0'][0]-tr, out['db1'][0]-tr, out['db2'][0]-tr, out['db0errm'][0], out['db1errm'][0], out['db2errm'][0])
    t1 = out['truth'][1]
    s += '   {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:8.6f} {:8.6f} {:8.6f}'.format(t1, out['db0'][1]-t1, out['db1'][1]-t1, out['db2'][1]-t1,
      out['db0errm'][1], out['db1errm'][1], out['db2errm'][1])
    t2 = out['truth'][2]
    s += '   {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:8.6f} {:8.6f} {:8.6f}'.format(t2, out['db0'][2]-t2, out['db1'][2]-t2, out['db2'][2]-t2,
      out['db0errm'][2], out['db1errm'][2], out['db2errm'][2])
    s += '   {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:9.6f}'.format(out['r0'][0,1], out['r1'][0,1], out['r2'][0,1],
      out['r0'][0,2], out['r1'][0,2], out['r2'][0,2])
    print(s)
    sys.stdout.flush()
