import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt

seaborn.set(rc={'figure.figsize':(15, 10),})
seaborn.set_context('talk') # or paper


# Define gauss function
def gaus(x, a, mu, sigma):
    return  a*np.exp(-(x-mu)**2/2./sigma**2)/np.sqrt(2./np.pi)/sigma


# Define fit function
def fit_func(bins, n, func):
    from scipy.optimize import curve_fit
    center = (bins[:-1] + bins[1:]) / 2
    popt, pcov = curve_fit(func, center, n, p0=(100, 100, 10))
    #print(" Fitted parameters: \n a [-]: \t ", popt[0],
    #      " \n \u03BC [MeV/c2]: \t ", popt[1],
    #      " \n \u03C3 [MeV/c2]: \t ", popt[2])
    print(" Fitted parameters: \n ",
          " \n a     [-]:      \t ", popt[0],
          " \n mu    [MeV/c2]: \t ", popt[1],
          " \n sigma [MeV/c2]: \t ", popt[2])

    #x = np.arange(0, 300, 1)
    #y = func(x, popt[0], popt[1], popt[2])
    #plt.plot(x, y, label='Fit: mass=%5.3f, width=%5.3f' % (popt[1], popt[2]))
    #plt.legend()

    return popt[0], popt[1], popt[2]


# Define histogram range and binning
x_min    = -10
x_max    = 390
n_bins_x = 80
x_bins = np.linspace(x_min,x_max,n_bins_x+1)


# Define parameters of the frame
fig = plt.figure() # plt.figure(figsize=(width,height))
#fig.patch.set_facecolor('white')
#fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111)
#ax.patch.set_facecolor('#ababab') # #ababab
ax.patch.set_alpha(0.0)
ax.spines['bottom'].set_color('0.5') #'black', ...
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_color('0.5')
ax.spines['top'].set_linewidth(2)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_color('0.5')
ax.spines['right'].set_linewidth(2)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_color('0.5')
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_visible(True)


# Ticks, grid and ticks labels
ax.tick_params(direction='in', length=10, width=2,                  # direction, length and width of the ticks (in, out, inout)
                colors='0.5',                                       # color of the ticks ('black', '0.5')
                bottom=True, top=True, right=True, left=True,       # whether to draw the respective ticks
                zorder = 10.,                                       # tick and label zorder
                pad = 10.,                                          # distance between ticks and tick labels
                labelsize = 17,                                     # size of the tick labels
                labelright=False, labeltop=False)                   # wether to draw the tick labels on axes
                #labelrotation=45.                                  # rotation of the labels
                #grid_color='black',                                # grid
                #grid_alpha=0.0,
                #grid_linewidth=1.0,
# colors='black','0.5'


# Define dataframes
#df = pd.read_csv(chain_cfg['name']+'_log.csv')
#print(df1.to_string())
df_00_fv14_th00 = pd.read_csv('00eV_threshold/pi0_chain_fiducialized_14px_10000ev_log.csv')
df_00_fv00_th35 = pd.read_csv('35eV_threshold/pi0_chain_fiducialized_0px_10000ev_log.csv')
df_00_fv14_th35 = pd.read_csv('35eV_threshold/pi0_chain_fiducialized_14px_10000ev_log.csv')

# Plot dataframes
n_0, bins_0, patches_0 = plt.hist(df_00_fv14_th00.pion_mass, bins=n_bins_x, range=[x_min,x_max], histtype='step', color='r', linewidth=3, alpha=0.7)
n_1, bins_1, patches_1 = plt.hist(df_00_fv00_th35.pion_mass, bins=n_bins_x, range=[x_min,x_max], histtype='step', color='g', linewidth=3, alpha=0.7)
n_1, bins_1, patches_1 = plt.hist(df_00_fv14_th35.pion_mass, bins=n_bins_x, range=[x_min,x_max], histtype='step', color='b', linewidth=3, alpha=0.7)

# Fit the peaks with Gaussians
# If you want to draw the fit function: uncomment lines in fit_func
#a_0, mu_0, sigma_0 = fit_func(bins_0, n_0, gaus)
#a_1, mu_1, sigma_1 = fit_func(bins_1, n_1, gaus)


# Legend
#mu_0    = float("{:.2f}".format(mu_0))
#sigma_0 = float("{:.2f}".format(sigma_0))
#mu_1    = float("{:.2f}".format(mu_1))
#sigma_1 = float("{:.2f}".format(sigma_1))
#entry_0 = 'fiducialized 0 px (\u03BC=' + str(mu_0) + ', \u03C3=' + str(sigma_0) + ')'
#entry_1 = 'fiducialized 14 px (\u03BC=' + str(mu_1) + ', \u03C3=' + str(sigma_1) + ')'
entry_0 = 'fiducialized 14 px, no threshold'
entry_1 = 'fiducialized 0 px, 35 MeV threshold'
entry_2 = 'fiducialized 14 px, 35 MeV threshold'
plt.legend([entry_0, entry_1, entry_2], loc=[0.5,0.85], prop={'size': 17}) # loc='upper right'


# Axis labels
plt.xlabel('Invariant $\pi^0$ mass [MeV/c$^2$]', fontsize=20, labelpad=20)
plt.ylabel('Entries [-]', fontsize=20, labelpad=20)


# Save figure
plt.savefig("pi0_mass_peak.png", dpi=400)
#plt.show()
