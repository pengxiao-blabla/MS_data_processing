# -------------------
# MY PYTHON VARIABLES
# ------------------
import numpy as np
from numpy import random
from scipy import special as spl
from scipy import interpolate as intp
pi = np.pi

# basic parameters_
n = 1.
wpe = 1.
c = 1.
me = 1.
e = 1.
mp = 100 * me

wpp = 1./np.sqrt(mp)
v_ratio = 15.        #c/vA


vA = c/v_ratio
wcp = wpp/v_ratio
lambda_i = c/wpp
B0 = wcp*mp/e


Lsim = [102.4 * lambda_i, 12.8* lambda_i]
Nx = 512
Ny = 128
dx = Lsim[0]/Nx      # dx = 0.5*lambda_i
dy = Lsim[1]/Ny      # dy = 0.1*lambda_i
dt = 0.001/wcp
tsim = 600/wcp
nsqrt = 40
nppc = nsqrt * nsqrt
number_cell = nppc * Nx * Ny

# --------------------------------------
# Dfine the position function and get the shell distribution random
# --------------------------------------

# shell distribution
ns = 0.2 * n
vs = 2.0*vA
vsth = 0.45*vA

def shell_cdf(v,vshell,vst):
    vst2=vst*vst
    vst3=vst2*vst
    b=vshell/vst
    a=(vshell-v)/vst

    cs=pi*vst3*(2*np.exp(-b*b)*b+np.sqrt(pi)*(2*b*b+1)*(1+spl.erf(b)))
    shellc=(pi*vst3/cs)*(2*np.exp(-b*b)-2*np.exp(-a*a)*(b+v/vst)+np.sqrt(pi)*(2*b*b+1)*(spl.erf(b)-spl.erf(a)))
    return shellc

#vphi random number generator
def inv_vphi_cdf(g):
    inv_vphi=np.arccos(1-2*g)
    return inv_vphi

u=random.uniform(0,1,size=number_cell)
vphi=inv_vphi_cdf(u)

#vtheta random number generator
vtheta=random.uniform(0,2*pi,size=number_cell)

#vx,y,z random number generator
vr=np.linspace(0,0.4,number_cell)
y1=shell_cdf(vr,vs,vsth)
shell_inv_cdf=intp.interp1d(y1,vr)

y1_new=random.uniform(0,1,size=number_cell)
vr_new=shell_inv_cdf(y1_new)

px=mp*vr_new*np.sin(vphi)*np.cos(vtheta)
py=mp*vr_new*np.sin(vphi)*np.sin(vtheta)
pz=mp*vr_new*np.cos(vphi)

momentum_init = np.array([px, py, pz])

del px,py,pz,vr_new

# electrons and background ions in the present study have a Maxwellian distribution
vbth = 0.03*vA
T = mp * vbth**2
Te = [T, T, T]

# --------------------------------------
# Initialize the position of the ion
# --------------------------------------


def position_init2D(Lx, Ly, nx, ny, nsqrt):
    dx = Lx / nx
    dy = Ly / ny
    ddx = dx / (nsqrt + 1)
    ddy = dy / (nsqrt + 1)

    # x方向上：
    x_index = np.zeros(nsqrt * nx)
    for i in range(nx):
        for j in range(nsqrt):
            x_index[i * nsqrt + j] = i * dx + j * ddx + ddx

    # y方向上：
    y_index = np.zeros(nsqrt * ny)
    for i in range(ny):
        for j in range(nsqrt):
            y_index[i * nsqrt + j] = i * dy + j * ddy

    position_x = np.tile(x_index, nsqrt * ny)
    position_y = np.repeat(y_index, nsqrt * nx)
    position_init = np.vstack((position_x, position_y))
    return position_init


position = position_init2D(Lsim[0], Lsim[1], Nx, Ny, nsqrt)
w_ion = np.full(int(number_cell), ns / nppc * dx * dy)
po_init_w = np.vstack((position, w_ion))

del position,w_ion
# --------------------------------------
# SMILEI's VARIABLES (DEFINED IN BLOCKS)
# --------------------------------------

Main(
    geometry="2Dcartesian",

    interpolation_order=2,

    timestep=dt,
    simulation_time=tsim,

    cell_length=[dx, dy],
    grid_length=Lsim,

    number_of_patches=[64, 16],

    EM_boundary_conditions=[
        ['periodic'],
        ['periodic']
    ],

    random_seed=smilei_mpi_rank
)

# APPLY EXTERNAL FIELD

ExternalField(
    field='Bx',
    profile=constant(B0)
)

Species(
    name='electron',
    position_initialization='regular',
    momentum_initialization='maxwell-juettner',
    particles_per_cell=nppc,
    mass=me,
    charge=-e,
    number_density=n,
    temperature=Te,
    boundary_conditions=[
        ['periodic', 'periodic'],
        ['periodic', 'periodic']
    ],
)

Species(
    name='cold_ion',
    position_initialization='regular',
    momentum_initialization='maxwell-juettner',
    particles_per_cell=nppc,
    mass=mp,
    charge=e,
    number_density=1 - ns,
    temperature=Te,
    boundary_conditions=[
        ['periodic', 'periodic'],
        ['periodic', 'periodic']
    ],
)

Species(
    name='shell_ion',
    position_initialization=po_init_w,
    momentum_initialization=momentum_init,
    # particles_per_cell = nppc,
    # tempertuare
    mass=mp,
    charge=e,
    boundary_conditions=[
        ['periodic', 'periodic'],
        ['periodic', 'periodic']
    ],
)

DiagScalar(
    every=500
)

DiagFields(
    every=50,
    fields=["Ex","Ez","Ey", "By","Bx","Bz"],
)

from numpy import s_
DiagFields(
    every=10,
    fields=["Ex","Ey","Ez","Bx","By","Bz"],
    subgrid=s_[256,64]
)

# cold ions
DiagParticleBinning(
    name = "Pxx_c",
    deposited_quantity = "weight_vx_px",
    every = 100,
    species = ['cold_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pyy_c",
    deposited_quantity = "weight_vy_py",
    every = 100,
    species = ['cold_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pzz_c",
    deposited_quantity = "weight_vz_pz",
    every = 100,
    species = ['cold_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "ntot_c",
    deposited_quantity = "weight",
    every = 100,
    species = ['cold_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

# shell ions
DiagParticleBinning(
    name = "Pxx_s",
    deposited_quantity = "weight_vx_px",
    every = 100,
    species = ['shell_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pyy_s",
    deposited_quantity = "weight_vy_py",
    every = 100,
    species = ['shell_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pzz_s",
    deposited_quantity = "weight_vz_pz",
    every = 100,
    species = ['shell_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "ntot_s",
    deposited_quantity = "weight",
    every = 100,
    species = ['shell_ion'],
    axes =  [["x",0,Lsim[0],1]]
)

# electron
DiagParticleBinning(
    name = "Pxx_e",
    deposited_quantity = "weight_vx_px",
    every = 100,
    species = ['electron'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pyy_e",
    deposited_quantity = "weight_vy_py",
    every = 100,
    species = ['electron'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "Pzz_e",
    deposited_quantity = "weight_vz_pz",
    every = 100,
    species = ['electron'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagParticleBinning(
    name = "ntot_e",
    deposited_quantity = "weight",
    every = 100,
    species = ['electron'],
    axes =  [["x",0,Lsim[0],1]]
)

DiagTrackParticles(
    species="cold_ion",
    every=[0,200000,100000],
    attributes=["px", "py", "pz"]
)

DiagTrackParticles(
    species="shell_ion",
    every=[0,200000,100000],
    attributes=["px", "py", "pz"]
)



