import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate


def f(y, x, params):
    r = y      # unpack current values of y
    eta, R, L = params  # unpack parameters
    derivs = 0.84/(eta*(3+eta**2))*(1/R*(1+L/((1+eta**2)*r))-1)      # list of dy/dt=f functions
    return derivs

# Parameters
eta = 3.35          # form factor
R =  5        # flux ratio
L = 1300     # diffusion length
tg = 30

#print(2*L/((R-1)*(1+eta**2)))
# Initial values
r0 = 15    # initial size

# Bundle parameters for ODE solver
params = [eta, R, L]

# Bundle initial conditions for ODE solver
y0 = r0

# Make time array for solution
xStop = 6800.
xInc = 100
x = np.arange(300., xStop, xInc)

############################### Call the ODE solver################################
psoln = integrate.odeint(f, y0, x, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,12))
# Plot diameter as a function of length
ax1 = fig.add_subplot(211)
ax1.plot(x, 2*psoln[:,0])
ax1.set_xlabel('Length')
ax1.set_ylabel('Diameter')
plt.ylim((0,120))
plt.text(4300,90,"Eta = " + repr(eta) + "\nFlux ratio = " + repr(R)+ "\nDiffusion length = " + repr(L)+" nm")
# Add VS growth
vsr = 0.405
ax =  76

rtotal = 2*psoln[:,0] + 2*vsr * (tg - x / ax)
ax1.plot(x, rtotal)

###############################steady state diameter#######################################
R_ss = np.arange(1.5,5,0.05)
r_ss=2*L/((R_ss-1)*(1+eta**2))

#ax2 = fig.add_subplot(312)
#ax2.plot(R_ss, r_ss)
#ax2.set_xlabel('Flux ratio')
#ax2.set_ylabel('Steady state diameter')

#experimental data
data_x = [2200, 6700, 300]
data_y = [44.7, 56, 30]
#data_x = [1900, 7000, 300]
#data_y = [81.7, 130, 30]


ax1.plot(data_x, data_y, 'or')

################ exponential VS growth rate #########

def VS_GR(t,v_0,v_ax,Lambda,y):
    y_total=v_ax*t
    if y<y_total:
        GR=v_0*(1-math.exp(-(v_ax*t-y)/Lambda))
        #GR=math.exp(-(v_ax*t-y)/Lambda)
        #GR=(v_ax*t-y)/Lambda
    else:
        GR=0
    return GR


VS_vec=np.vectorize(VS_GR)

################export instant growth rate at end of growth############
fname='InstantGR'+repr(tg)+'.txt'
GR_instant=VS_vec(tg,0.478,76,1150,x)
np.savetxt(fname,np.transpose([x,GR_instant]),delimiter=',')

#############integrate over growth rate to get diameter################
VS_x=[]
for x_i in x:
    VS_int = integrate.quad(VS_vec,0,tg,args=(0.52,76,1200,x_i))
    print(repr(x_i/xStop*100)+ ' %')
    #VS_x=VS_x.append(VS_int[0])
    #VS_graph=VS_graph.append(5)
    VS_x[len(VS_x):] = [VS_int[0]]

#ax2 = fig.add_subplot(312)
ax1.plot(x, VS_x)        

rtotal_exp = 2*psoln[:,0] + VS_x + VS_x
ax1.plot(x, rtotal_exp)
ax1.plot(x, vsr * (tg - x / ax))
fname='totalExpDiameter'+repr(tg)+'.txt'
np.savetxt(fname,np.transpose([x,rtotal_exp]),delimiter=',')
fname='VLSDiameter'+repr(tg)+'.txt'
np.savetxt(fname,np.transpose([x,2*psoln[:,0]]),delimiter=',')

position=[6511,5887,5264,4448,4000,3484,2801,2271,279,1060,1899,2657,3367,4339]
diameter=[53.6,63.5,63.5,73.4,87.4,83.3,91.3,91.3,97.2,97.2,97.2,91.3,90.1,83.3]
ax1.plot(position, diameter, 'og')
ax1.plot(x, rtotal)

##################GR at bottom for different growth times##############

t= np.arange(6., 300, 1)
Diff=[200,500,1200,2000]

for Lambda in Diff:
    VS_t=[]
    print(repr(Lambda))
    for t_i in t:
        #divide grown thickness by growth time, to get linearized GR
        VS_tint = (integrate.quad(VS_vec,5,t_i,args=(0.475,76,Lambda,300)))/(t_i-5) 
        VS_t[len(VS_t):] = [VS_tint[0]]
    
    
    ax3 = fig.add_subplot(212)
    ax3.plot(t,VS_t)

    time=[30,88]
    GR_t=[0.292,0.422]
    GR_t2=[0.415,0.445]
    ax3.plot(time,GR_t,'or')
    ax3.plot(time,GR_t2,'og')
    ax3.set_xlabel('growth time')
    ax3.set_ylabel('effective growth rate \n at bottom')
    fn='GRvst'+repr(Lambda)+'.txt'
    np.savetxt(fn,np.transpose([t,VS_t]),delimiter=',')
    
plt.text(80,0.1,"Diffusion length (nm)\n" + repr(Diff))
ax3.axhline(0.52,linestyle='--')
plt.ylim((0,0.65))
plt.show()
