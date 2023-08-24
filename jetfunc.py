# Functions for jet.py

import yoda
#import rivet
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import curve_fit
import os



def RAA(yodaPbPb, yodapp, obs, nrebin = 1, sigmann = 67.6, verbose = False):
    '''
    Calculates R_{AA} given the yoda files for pp and PbPb comparing the
    observable obs for each.
    (str, str, str, int, bool) -> (dict)
    '''

    # Read files
    histos_pp = yoda.read(yodapp)
    histos_PbPb = yoda.read(yodaPbPb)
    pp_jet = histos_pp[obs]
    PbPb_jet = histos_PbPb[obs]

    # Prepare for comparison
    pp_jet.rebinBy(nrebin)
    pp_evtc = histos_pp['/_EVTCOUNT'].sumW()
    pp_xsec = histos_pp['/_XSEC'].point(0).x
    pp_jet.scaleW(pp_xsec / pp_evtc)

    PbPb_jet.rebinBy(nrebin)
    PbPb_evtc = histos_PbPb['/_EVTCOUNT'].sumW()
    PbPb_xsec = histos_PbPb['/_XSEC'].point(0).x
    PbPb_jet.scaleW(PbPb_xsec / PbPb_evtc)

    if verbose:
        print('Cross-section rescaling: ' + str(PbPb_xsec * pp_evtc / (pp_xsec * PbPb_evtc)))

    # R_AA calulation
    raa = PbPb_jet / pp_jet
    x = np.asarray(raa.xVals())
    y = np.asarray(raa.yVals())
    yerr = np.asarray((raa.yMaxs() - raa.yMins()) / 2)
    xerr = np.asarray((raa.xMaxs() - raa.xMins()) / 2)

    # Propagate the error from the sums of weights
    # Depending on the number of events, this step has no significant impact
    Serr_pp = histos_pp['/_EVTCOUNT'].relErr
    Serr_PbPb = histos_PbPb['/_EVTCOUNT'].relErr
    raaerr = np.sqrt(y ** 2 * (Serr_pp ** 2  + Serr_PbPb ** 2) + yerr ** 2)


    return [x, y, raaerr, xerr]

def RAARebinTo(yodaPbPb, yodapp, obs, rebinto, verbose = False):
    '''
    Calculates R_{AA} given the yoda files for pp and PbPb comparing the
    observable obs for each, rebinning the histograms to rebinto.
    (str, str, str, list, bool) -> (dict)
    '''

    # Read files
    histos_pp = yoda.read(yodapp)
    histos_PbPb = yoda.read(yodaPbPb)
    pp_jet = histos_pp[obs]
    PbPb_jet = histos_PbPb[obs]

    # Prepare for comparison
    pp_jet.rebinTo(rebinto)
    pp_evtc = histos_pp['/_EVTCOUNT'].sumW()
    pp_xsec = histos_pp['/_XSEC'].point(0).x
    pp_jet.scaleW(pp_xsec / pp_evtc)

    PbPb_jet.rebinTo(rebinto)
    PbPb_evtc = histos_PbPb['/_EVTCOUNT'].sumW()
    PbPb_xsec = histos_PbPb['/_XSEC'].point(0).x
    PbPb_jet.scaleW(PbPb_xsec / PbPb_evtc)

    if verbose:
        print('Cross-section rescaling: ' + str(PbPb_xsec * pp_evtc / (pp_xsec * PbPb_evtc)))

    # R_AA calulation
    raa = PbPb_jet / pp_jet
    x = np.asarray(raa.xVals())
    y = np.asarray(raa.yVals())
    yerr = np.asarray((raa.yMaxs() - raa.yMins()) / 2)
    xerr = np.asarray((raa.xMaxs() - raa.xMins()) / 2)

    # Propagate the error from the sums of weights
    # Depending on the number of events, this step has no significant impact
    Serr_pp = histos_pp['/_EVTCOUNT'].relErr
    Serr_PbPb = histos_PbPb['/_EVTCOUNT'].relErr
    raaerr = np.sqrt(y ** 2 * (Serr_pp ** 2  + Serr_PbPb ** 2) + yerr ** 2)


    return [x, y, raaerr, xerr]

def RAARebin(yodaPbPb, yodapp, obs, binmin, binmax, nrebin = 1):
    # Read files
    histos_pp = yoda.read(yodapp)
    histos_PbPb = yoda.read(yodaPbPb)
    pp_jet = histos_pp[obs]
    PbPb_jet = histos_PbPb[obs]

    # Prepare for comparison
    pp_jet.rebinBy(nrebin)
    pp_evtc = histos_pp['/_EVTCOUNT'].sumW()
    pp_xsec = histos_pp['/_XSEC'].point(0).x
    pp_jet.scaleW(pp_xsec / pp_evtc)

    PbPb_jet.rebinBy(nrebin)
    PbPb_evtc = histos_PbPb['/_EVTCOUNT'].sumW()
    PbPb_xsec = histos_PbPb['/_XSEC'].point(0).x
    PbPb_jet.scaleW(PbPb_xsec / PbPb_evtc)

    print('Cross-section rescaling: ' + str(PbPb_xsec * pp_evtc / (pp_xsec * PbPb_evtc)))

    # Rebinning
    edges = [71., 79., 89., 100., 126., 158., 200., 251., 300., 350., 450., 630., 1000.]
    # edges = [71., 79., 89., 100., 126., 158., 200., 251., 316., 398., 500., 650., 1000.]
    newedges = edges[:binmin + 1] + edges[binmax:]
    PbPb_jet.rebinTo(newedges)
    pp_jet.rebinTo(newedges)

    # R_AA calulation
    raa = PbPb_jet / pp_jet

    x = np.asarray(raa.xVals())
    y = np.asarray(raa.yVals())
    yerr = np.asarray((raa.yMaxs() - raa.yMins()) / 2)
    xerr = np.asarray((raa.xMaxs() - raa.xMins()) / 2)

    # Propagate the error from the sums of weights
    # Depending on the number of events, this step has no significant impact
    Serr_pp = histos_pp['/_EVTCOUNT'].relErr
    Serr_PbPb = histos_PbPb['/_EVTCOUNT'].relErr
    raaerr = np.sqrt(y ** 2 * (Serr_pp ** 2  + Serr_PbPb ** 2) + yerr ** 2)

    ptrange = '{0:.0f} GeV $< p_T <$ {1:.0f} GeV'.format(edges[binmin], edges[binmax])

    return [x, y, raaerr, xerr], ptrange


# Vn fit functions
def v2Fit(x, vn, A):
    '''
    Fit function for the v2 atlas analysis
    '''
    return A * (1 + 2 * vn * np.cos(2 * x))



def v3Fit(x, vn, A):
    '''
    Fit function for the v3 atlas analysis
    '''
    return A * (1 + 2 * vn * np.cos(3 * x))



def v4Fit(x, vn, A):
    '''
    Fit function for the v4 atlas analysis
    '''
    return A * (1 + 2 * vn * np.cos(4 * x))



def TextBoxInfo(cent = 0, R = 0.4, extra = 0, tc = 0, mds = 0, model = 2, energy = 5.02, lead = 1, pos = 'left'):
    ''' Function to write information on plots easily '''

    if model == 0:
        txtinfo = r'JEWEL+PYTHIA' + '\n'
    elif model == 1:
        txtinfo = r'JEWEL2.2+PYTHIA Glauber+Bjorken' + '\n'
    elif model == 2:
        txtinfo = r'JEWEL2.2+PYTHIA $\rm T_RENTo$+vUSP' + '\n'
    elif model == 3:
        txtinfo = r'JEWEL+PYTHIA $\rm T_RENTo$' + '\n'
    elif model == 4:
        txtinfo = r'JEWEL2.2 $\rm T_RENTo$ + vUSPhydro' + '\n'
    else:
        txtinfo = r'JEWEL+PYTHIA MC-KLN+vUSP' + '\n'

    if lead == 1:
        txtinfo += 'PbPb '
    txtinfo += r'$\sqrt{s_{NN}}$ = ' + str(energy) + ' TeV'

    if cent != 0:
        txtinfo += r' ' + cent + '\%'

    if tc != 0 and mds != 0:
        txtinfo += '\n' + r'$T_C$ = ' + str(tc) + r', MDS = ' + str(mds)
    elif tc == 0 and mds != 0:
        txtinfo += '\n' + r'MDS = ' + str(mds)
    elif tc != 0 and mds == 0:
        txtinfo += '\n' + r'$T_C$ = ' + str(tc)

    if R != 0:
        txtinfo += '\n' + r'Anti-$k_t$ R = ' + str(R)
    # else:
    #     txtinfo += '\n' + r'Anti-$k_t$'

    if extra != 0:
        txtinfo += '\n' + extra

    print('Text box info: ' + txtinfo)
    if pos == 'left':
        dx = plt.gca().get_xlim()[0] + (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * 0.015
        dy = plt.gca().get_ylim()[1] - (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * 0.03

    elif pos == 'right':
        dx = plt.gca().get_xlim()[0] + (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * 1.015
        dy = plt.gca().get_ylim()[1] - (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * 0.03

    plt.text(dx, dy, txtinfo, horizontalalignment=pos, verticalalignment='top', usetex=True, fontsize=12, ma='left')
    # return [dx, dy, txtinfo]



def calculateVNATLAS(yodafile, n, R = 0.2, aconst = False, nrebin = 8):
    '''
    Calculate vn using the ATLAS method from a yoda file.
    If aconst is true, return normalization constants instead of vns
    (str, int, bool) -> (list)
    '''

    histos = yoda.read(yodafile)
    sn = str(n)

    atlas_obs = ['/USPJWL_JETSPEC/71_79_v', '/USPJWL_JETSPEC/79_89_v', '/USPJWL_JETSPEC/89_100_v',
    '/USPJWL_JETSPEC/100_126_v', '/USPJWL_JETSPEC/126_158_v', '/USPJWL_JETSPEC/158_200_v',
    '/USPJWL_JETSPEC/200_251_v', '/USPJWL_JETSPEC/251_300_v', '/USPJWL_JETSPEC/300_350_v',
    '/USPJWL_JETSPEC/350_450_v', '/USPJWL_JETSPEC/450_630_v', '/USPJWL_JETSPEC/630_1000_v']
    obs = [s + str(n) + '_R' + str(R) for s in atlas_obs]

    vn, vnerr, a, aerr = [], [], [], []

    for i in obs:
        h = histos[i]
        h.normalize(includeoverflows=False)
        h.rebin(nrebin)
        x = h.xVals()
        y = h.yVals()
        yerr = (h.yMaxs() - h.yMins()) / 2

        if n == 2:
            params, pcov = curve_fit(v2Fit, x, y, sigma=yerr)

        elif n == 3:
            params, pcov = curve_fit(v3Fit, x, y, sigma=yerr)

        else:
            params, pcov = curve_fit(v4Fit, x, y, sigma=yerr)

        vn.append(params[0])
        vnerr.append(np.sqrt(pcov[0, 0]))
        a.append(params[1])
        aerr.append(pcov[1, 1])

    # Return vn values, vn errors, and normalization constant values
    if aconst == False:
        return [np.asarray(vn), np.asarray(vnerr)]
    else:
        return [np.asarray(a), np.asarray(aerr)]



def calculateVNALICE(yodafile, n, nrebin = 1):
    '''
    Calculate v_n using the generalized ALICE method from a yoda file
    (str, int) -> (list)
    '''

    histos_exec = yoda.read(yodafile)

    if n == 2:
        jets_in = histos_exec['/JET_VN/JetsInPlane_v2']
        jets_out = histos_exec['/JET_VN/JetsOutOfPlane_v2']

    elif n == 3:
        jets_in = histos_exec['/JET_VN/JetsInPlane_v3']
        jets_out = histos_exec['/JET_VN/JetsOutOfPlane_v3']

    else:
        jets_in = histos_exec['/JET_VN/JetsInPlane_v4']
        jets_out = histos_exec['/JET_VN/JetsOutOfPlane_v4']




    numerator = jets_in - jets_out
    denominator = jets_in + jets_out
    numerator.rebinBy(nrebin)
    denominator.rebinBy(nrebin)

    vn = numerator / denominator
    vn.scaleY(np.pi / 4)
    x = vn.xVals()
    y = vn.yVals()
    xerr = (vn.xMaxs() - vn.xMins()) / 2
    yerr = (vn.yMaxs() - vn.yMins()) / 2

    # Same order as plt.errorbar()
    return [np.asarray(x), np.asarray(y), np.asarray(yerr), np.asarray(xerr)]



def XJ(yodafile, ptrange = 0, R = 0.4, nrebin = 1):
    '''
    Calculate dijet asymetry x_J for a certain p_T (of leading) range from a yodafile
    (str, int, int) -> (list)
    '''
    hists = yoda.read(yodafile)

    latex_range = [r'10 GeV $< p_{T,1} <$ 30 GeV', r'30 GeV $< p_{T,1} <$ 60 GeV',
                   r'60 GeV $< p_{T,1} <$ 90 GeV', r'90 GeV $< p_{T,1} <$ 120 GeV',
                   r'120 GeV $< p_{T,1} <$ 158 GeV', r'158 GeV $< p_{T,1} <$ 178 GeV',
                   r'100 GeV $< p_{T,1} <$ 126 GeV', r'112 GeV $< p_{T,1} <$ 126 GeV',
                   r'178 GeV $< p_{T,1} <$ 200 GeV', r'200 GeV $< p_{T,1} <$ 224 GeV',
                   r'224 GeV $< p_{T,1} <$ 251 GeV', r'251 GeV $< p_{T,1} <$ 282 GeV',
                   r'282 GeV $< p_{T,1} <$ 316 GeV', r'316 GeV $< p_{T,1} <$ 398 GeV',
                   r'398 GeV $< p_{T,1} <$ 562 GeV', r'562 GeV $< p_{T,1} < $ 700 GeV',
                   r'700 GeV $< p_{T,1} < $ 1000 GeV']

    xj_obs = ['xJ_10_30_R', 'xJ_30_60_R', 'xJ_60_90_R', 'xJ_90_120_R', 'xJ_120_158_R',
              'xJ_158_178_R', 'xJ_100_112_R', 'xJ_112_126_R', 'xJ_178_200_R', 
              'xJ_200_224_R', 'xJ_224_251_R', 'xJ_251_282_R', 'xJ_282_316_R', 
              'xJ_316_398_R', 'xJ_398_562_R',
              'xJ_562_700_R', 'xJ_700_1000_R']
    if ptrange == 6:
        obs = ['/USPJWL_JETSPEC/' + xj_obs[6] + str(R), '/USPJWL_JETSPEC/' + xj_obs[7] + str(R)]
        h1 = hists[obs[0]]
        h1.rebinBy(nrebin)
        h1.normalize(includeoverflows=False)
        
        h2 = hists[obs[1]]
        h2.rebinBy(nrebin)
        h2.normalize(includeoverflows=False)
        
        x = h2.xVals()
        y = (h1+h2).yVals()
        y = y/(h1.sumW()+h2.sumW())
        xerr = (((h1.xMaxs() - h1.xMins()) / 2)**2 + ((h2.xMaxs() - h2.xMins()) / 2)**2)**(1/2)
        yerr = (((h1.yMaxs() - h1.yMins()) / 2)**2 + ((h2.yMaxs() - h2.yMins()) / 2)**2)**(1/2)
    
    else:
        obs = ['/USPJWL_JETSPEC/' + i + str(R) for i in xj_obs]

        h = hists[obs[ptrange]]
        h.rebinBy(nrebin)
        h.normalize(includeoverflows=False)  
        x = h.xVals()
        y = h.yVals()
        xerr = (h.xMaxs() - h.xMins()) / 2
        yerr = (h.yMaxs() - h.yMins()) / 2
        
        
        

    return [np.asarray(x), np.asarray(y), np.asarray(yerr), np.asarray(xerr), latex_range[ptrange]]



def HistQuotient(y1, y1err, y2, y2err, uncert = 0):
    '''
    Given two numpy arrays and its errors, return the quotient y2 / y1 and
    the propagated error with method defined by uncert
    (0 = Taylor, 1 = Highest dominates).
    (np.array, np.array, np.array, np.array) -> (np.array)
    '''

    q = y2 / y1
    if uncert == 0:
        qerr = np.sqrt((y2err / y1) ** 2 + (y1err * q / y1) ** 2)

    elif uncert == 1:
        per_err1 = np.absolute(y1err / y1)
        per_err2 = np.absolute(y2err / y2)
        qerr = q * np.maximum(per_err1, per_err2) # Max element by element
    else:
        qerr = 0
    return [q, qerr]


def VNRebin(yodafile, n, R = 0.2, minobs = 0, maxobs = 11, nrebin = 8):
    '''
    Calculates the vn(R) of a yodafile for a integrated pT interval, which is
    determined by the number of merged histogram between [minobs, maxobs[
    (str, int, float, int, int) -> (list, str)
    '''

    atlas_obs = ['/USPJWL_JETSPEC/71_79_v', '/USPJWL_JETSPEC/79_89_v', '/USPJWL_JETSPEC/89_100_v',
    '/USPJWL_JETSPEC/100_126_v', '/USPJWL_JETSPEC/126_158_v', '/USPJWL_JETSPEC/158_200_v',
    '/USPJWL_JETSPEC/200_251_v', '/USPJWL_JETSPEC/251_300_v', '/USPJWL_JETSPEC/300_350_v',
    '/USPJWL_JETSPEC/350_450_v', '/USPJWL_JETSPEC/450_630_v', '/USPJWL_JETSPEC/630_1000_v']
    obs = [s + str(n) + '_R' + str(R) for s in atlas_obs]

    # Determine pT range
    minpt = atlas_obs[minobs].split('/')[-1].split('_')[0]
    maxpt = atlas_obs[maxobs - 1].split('/')[-1].split('_')[1]
    print('Rebinning v{0} data from {1} to {2} GeV, for {3}'.format(n, minpt, maxpt, yodafile))
    ptrange = r'{0} GeV $< p_T <$ {1} GeV'.format(minpt, maxpt)


    # Load file
    file = yoda.read(yodafile)
    bins = file[obs[0]].xEdges()

    # Fill new histogram
    newh = yoda.Histo1D(bins, '/USPJWL_JETSPEC/All_v' + str(n) + '_R' + str(R))
    for i in obs[minobs:maxobs]:
        h = file[i]
        newh += h

    newh.rebin(nrebin)
    x = newh.xVals()
    y = newh.yVals()
    yerr = (newh.yMaxs() - newh.yMins()) / 2

    # Calculate vn
    if n == 2:
        params, pcov = curve_fit(v2Fit, x, y, sigma=yerr)

    elif n == 3:
        params, pcov = curve_fit(v3Fit, x, y, sigma=yerr)

    else:
        params, pcov = curve_fit(v4Fit, x, y, sigma=yerr)

    # Return [integrated vn, error] and latex ready pT range
    return [params[0], np.sqrt(pcov[0, 0])], ptrange



def WriteMedium(xgrid, ygrid, temperature, propertime, filename):
    ''' Append the medium profile for JEWEL's reader routine, given the x-y grid,
    temperature for each point and propertime, to the filename.

    (np.array, np.array, float, str) -> (None)
    '''

    # Check if file exists
    if os.path.exists(filename):
        write_condition = 'a'
    else:
        write_condition = 'w'

    file = open(filename, write_condition)

    for j in range(temperature[:][0].size):
        x = xgrid[j]

        for i in range(temperature[0].size):
            y = ygrid[i]

            T = temperature[i][j]
            if T > 0.1:
                file.write('{0} {1:.6f} {2:.6f} {3:.6f}\n'.format(propertime, x, y, T))



def ReadMedium(propertime, profile, tc, flow = 1):
    '''
    Reads the medium evolution profile and returns the temperature and local
    velocity profiles (x, y, T, ux, uy) in a given propertime.
    Ignore points where temperature is lower than tc.

    (float, np.array, float) -> (np.array, np.array, np.array, np.array, np.array)
    '''

    # Don't start if propertime is before taui or after tauf
    taui = profile[0][0]
    tauf = profile[-1][0]
    if propertime < taui or propertime > tauf:
        return None

    X = []
    Y = []
    T = []
    UX = []
    UY = []

    for l in profile:
        # Each line is written as [tau x y temp ux uy]
        if l[0] == propertime and l[3] >= tc:
            X.append(l[1])
            Y.append(l[2])
            T.append(l[3])
            if flow == 1:
                UX.append(l[4])
                UY.append(l[5])


        # If pass the propertime, ignore the rest of file
        elif l[0] > propertime:
            break

    return np.asarray(X), np.asarray(Y), np.asarray(T), np.asarray(UX), np.asarray(UY)



def ChiSquared(observable, experiment):
    '''
    Calculate the Chi^2 sum value and for a comparisson between the simulated results
    for the observable and experimental data. Assume the same x binning.

    (list(np.array), list(np.array)) -> (float)
    '''

    # Check sizes, remember that array: [x, y, yerr, xerr or nothing]
    if observable[1].size != experiment[1].size:
        print("Incompatible observable and experiment arrays")
        return None


    chiarray = (experiment[1] - observable[1]) ** 2 / np.abs(experiment[2] ** 2 + observable[2] ** 2)
    chi2 = np.sum(chiarray)

    if chi2 > 0.:
        return chi2

    else:
        print("Impossible value for chi2")
        return None



def DataFromExperiment(file, scatter, min = 0, max = 0):
    '''
    Grab data from the Yoda.Scatter2D named scatter from the file yoda, following
    the method that colaborations (ALICE, ATLAS, CMS) save them.

    (str, str) -> (list(np.array))
    '''
    data = yoda.read(file)[scatter]

    if max == 0:
        max = len(data.xVals()) + 1

    x = np.asarray(data.xVals())[min:max]
    y = np.asarray(data.yVals())[min:max]
    err = np.asarray((data.yMaxs() - data.yMins()) / 2)[min:max]
    errx = np.asarray((data.xMaxs() - data.xMins()) / 2)[min:max]
    return [x, y, err, errx]



def SimpleSquareDist(x1, y1, x2, y2, yerr):
    ''' Simple calculator of square distance between two set of points '''

    p = np.poly1d(np.polyfit(x2, y2, 3))
    ds = 0
    for i in range(len(y1)):
        ds += ((y1[i] - p(x1)[i]) / yerr[i]) ** 2

    return np.sqrt(ds / len(y1))



def CommatoPoint(title):
    '''
    Swaps all commas to points in file named title
    (str) -> None
    '''

    read = open(title, 'r')
    data = read.read().replace(',', '.')
    file = open(title, 'w')
    file.write(data)
    print(title + ': , -> .')



# New psi^{jet} and vn^{jet} functions
def AllSymmetryPlane(psi, n):
    allpsi = []
    for subn in range(n):
        psi_prime = psi + subn * 2 * np.pi / n
        psi_prime = AngleTo_0_2PI(psi_prime)
        allpsi.append(psi_prime)

    return allpsi


def AngleTo_0_2PI(angle):
    if angle <= 0:
        return angle + 2 * np.pi
    elif angle > 2 * np.pi:
        return angle - 2 * np.pi

    return angle


def AngleDistance(angle1, angle2):
    diff = np.abs(angle1 - angle2)

    if diff > np.pi:  # Check other side
        return 2 * np.pi - diff

    return diff


def psiJets(x, y, n, verbose = False):
    '''
    Calculate the psi^{jet}_n given a certain distribution (x, y)

    (np.array, np.array, int, bool) -> (float)
    '''

    num = 0
    den = 0
    for i in range(len(x)):
        num += np.sin(n * x[i]) * y[i]
        den += np.cos(n * x[i]) * y[i]

    # Psi is defined in [-pi, pi], need to transform it to [0, 2pi]
    cand1 = AngleTo_0_2PI(np.arctan(num / den) / n)
    cand2 = AngleTo_0_2PI((np.arctan(num / den) + np.pi) / n)

    vn1 = calculateVN(x, y, n, cand1)[0][0]
    vn2 = calculateVN(x, y, n, cand2)[0][0]

    if verbose:
        print(f'Calculating Psi_{n}')
        print(f'Cand1 = {cand1:.3f} => v{n} = {vn1:.3f}')
        print(f'Cand2 = {cand2:.3f} => v{n} = {vn2:.3f}')

    # Selection strategy, neFed for low number of events
    if vn1 > vn2:
        if verbose:
            print('Cand1 was chosen!\n')
        return cand1
    else:
        if verbose:
            print('Cand2 was chosen!\n')
        return cand2


def calculateVN(x, y, n, psi, nbins = 8):
    '''
    Calculate the v_n given a certain distribution (x, y) and a symmetry angle
    psi.

    (np.array, np.array, int, float, int) -> (list, list, float)
    '''

    # Create a histogram for distances
    deltaphi = yoda.Histo1D(nbins, 0, np.pi, f'Delta_phi')
    allpsi = AllSymmetryPlane(psi, n)

    for j in range(len(x)):
        dists = [AngleDistance(x[j], p) for p in allpsi]
        deltaphi.fill(n * np.min(dists), y[j])

    deltax = deltaphi.xVals()
    deltay = deltaphi.yVals()
    deltasigma = (deltaphi.yMaxs() - deltaphi.yMins()) / 2
    deltaw = deltaphi.sumW()  # This is the weight of the event

    params, pcov = curve_fit(vnFit, deltax, deltay, sigma=deltasigma, p0=[0., 1.])
    vnjet = [params[0], np.sqrt(pcov[0, 0])]
    normconst = [params[1], np.sqrt(pcov[1, 1])]

    return vnjet, normconst, deltaw


def vnFit(x, vn, A):
    '''
    Fit function for the vn atlas analysis
    '''
    return A * (1 + 2 * vn * np.cos(x))


def ReadNPZ(npzs, n, pt, opt = 0):
    '''
    Load the arrays of vn^jet, vn^soft, cos(n * delta psi) and
    the weights of each histogram given n, the pt bin and a list
    of npz files.

    (list, int, int) -> (np.array, np.array, np.array, np.array, np.array)
    '''
    data = []
    for f in npzs:
        data.append(np.load(f, allow_pickle=True))

    vnsoft = np.concatenate([x['vns'] for x in data])
    psisoft = np.concatenate([x['psis'] for x in data])
    weightsoft = np.concatenate([x['ws'] for x in data])

    if opt == 0:  # Caio's method
        vnjet = np.concatenate([x['vnjavg'] for x in data])[:, pt]
        psijet = np.concatenate([x['psijavg'] for x in data])[:, pt]
    else:  # Virginia's method
        vnjet = np.concatenate([x['vnj'] for x in data])[:, pt]
        psijet = np.concatenate([x['psij'] for x in data])[:, pt]
        print('WARNING: VIRGINIA METHOD ON VNS!')

    weightjet = np.concatenate([x['wj'] for x in data])[:, pt]
    numberjets = np.concatenate([x['nj'] for x in data])[:, pt]
    mod = np.concatenate([x['mod'] for x in data])[:, pt]

    cosdelta = np.cos(n * np.asarray(psijet - psisoft))
    weights = weightjet * weightsoft

    return vnjet, vnsoft, cosdelta, weights, numberjets, mod, psijet, psisoft

def VNJet(x):
    x = np.asarray(x)
    return np.average(x[0], weights=x[1])

def VNExp(x):
    x = np.asarray(x)
    vnjet = x[0]
    vnsoft = x[1]
    cosdelta = x[2]
    weights = x[3]

    den = np.average(vnsoft * vnjet * cosdelta, weights=weights)
    num = np.sqrt(np.average(vnsoft ** 2, weights=weights))

    return den / num

def ratioVNExp(x):
    x = np.asarray(x)
    den = [x[0], x[1], x[2], x[3]]
    num = [x[4], x[5], x[6], x[7]]
    # weights = x[2]

    denvn = VNExp(den)
    numvn = VNExp(num)
    ratio = denvn / numvn

    return ratio

# def ratioVNExp2(x):
#     x = np.asarray(x)
#     lenx = len(x[0])
#     den = x[:4]
#     num = x[4:]
#
#     return np.average(den) / np.average(num)


def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    x = np.asarray(x)
    n = len(x[0])
    idx = np.arange(n)
    return np.sum(func(x[:, idx != i]) for i in range(n)) / n

def jackknife_var(x, func):
    """Jackknife estiamte of the variance of the estimator func."""
    x = np.asarray(x)
    n = len(x[0])
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return (n - 1) / n * np.sum((func(x[:, idx != i]) - j_est) ** 2
                                    for i in range(n))


# def vnall(npz, n, exp = True, method = 0):
#     vn = []
#     vnerr = []
#
#     for pt in range(14):
#         alldata = ReadNPZ(npz, n, pt, method)
#         vnjet = alldata[0]
#         vnsoft = alldata[1]
#         cosdelta = alldata[2]
#         weights = alldata[3]
#         info1 = [vnjet, weights]
#         info2 = [vnjet, vnsoft, cosdelta, weights]
#
#         if exp == True:
#             vn.append(VNExp(info2))
#             vnerr.append(np.sqrt(jackknife_var(info2, VNExp)))
#
#         else:
#             vn.append(VNJet(info1))
#             vnerr.append(np.sqrt(jackknife_var(info1, VNJet)))
#
#     return [vn, vnerr]


def FilterVNInfo(x, verbose = False):
    x = np.asarray(x)
    copyx = np.copy(x)
    jet = copyx[0]
    soft = copyx[1]
    cos = copyx[2]
    we = copyx[3]

    original_len = len(jet)
    tobedeleted = []

    # Filter
    for i in range(original_len):
        # filter_vn = np.abs(jet[i] - soft[i]) / np.std(soft) > 3
        filter_vn = np.abs(jet[i]) > 0.5
        weights_rem = np.delete(np.copy(we), i)
        filter_w = np.abs(we[i] - np.mean(weights_rem)) / np.std(weights_rem) > 3
        if filter_vn or filter_w:
            tobedeleted.append(i)

            # if verbose:
            #     print(f'Medium {i} was filtered given weights')

    jet = np.delete(jet, tobedeleted)
    soft = np.delete(soft, tobedeleted)
    cos = np.delete(cos, tobedeleted)
    we = np.delete(we, tobedeleted)

    if original_len - len(jet) > 0.68 * original_len or verbose:
        print(f'WARNING: {100 * (original_len - len(jet)) / original_len:.0f}% of media have been filtered!')

    # if verbose:
    #     print(f'Number of filtered media: {original_len - len(vnjet)}')

    return [jet, soft, cos, we]

def vnall(npz, n, exp = True, method = 0, filter = True):
    vn = []
    vnerr = []

    for pt in range(14):
        alldata = ReadNPZ(npz, n, pt, method)
        vnjet = alldata[0]
        vnsoft = alldata[1]
        cosdelta = alldata[2]
        weights = alldata[3]

        if filter:
            filtered_data = FilterVNInfo(alldata[:4], False)
            vnjet = filtered_data[0]
            vnsoft = filtered_data[1]
            cosdelta = filtered_data[2]
            weights = filtered_data[3]

        info1 = [vnjet, weights]
        info2 = [vnjet, vnsoft, cosdelta, weights]

        if exp == True:
            vn.append(VNExp(info2))
            vnerr.append(np.sqrt(jackknife_var(info2, VNExp)))

        else:
            vn.append(VNJet(info1))
            vnerr.append(np.sqrt(jackknife_var(info1, VNJet)))

    return [vn, vnerr]


def JetSpectrum(yodafile, obs, normval = 5.6, nrebin = 2):
    histos = yoda.read(yodafile)
    spec = histos[obs]

    evtc = histos['/_EVTCOUNT'].sumW()
    xsec = histos['/_XSEC'].point(0).x * 1E-3

    spec.scaleW(xsec / (evtc * normval))

    x = np.asarray(spec.xVals())
    y = np.asarray(spec.yVals())
    yerr = np.asarray(spec.yErrs())
    xerr = np.asarray((spec.xMaxs() - spec.xMins()) / 2)

    return [x, y, yerr, xerr]



# Deprecated
# def RAA(yodaPbPb, yodapp, obs, nrebin = 1, sigmann = 67.6, verbose = False):
#     '''
#     Calculates R_{AA} given the yoda files for pp and PbPb comparing the
#     observable obs for each. Nucleon-nucleon cross section in mb.
#     (str, str, str, int, double, bool) -> (dict)
#     '''
#
#     # Read files
#     histos_pp = yoda.read(yodapp)
#     histos_PbPb = yoda.read(yodaPbPb)
#     pp_jet = histos_pp[obs]
#     PbPb_jet = histos_PbPb[obs]
#
#     # Prepare for comparison
#     pp_jet.rebinBy(nrebin)
#     pp_evtc = histos_pp['/_EVTCOUNT'].sumW()
#     pp_xsec = histos_pp['/_XSEC'].point(0).x
#     pp_jet.scaleW(pp_xsec / pp_evtc)
#
#     PbPb_jet.rebinBy(nrebin)
#     PbPb_evtc = histos_PbPb['/_EVTCOUNT'].sumW()
#     PbPb_xsec = histos_PbPb['/_XSEC'].point(0).x
#     PbPb_jet.scaleW(PbPb_xsec / (PbPb_evtc * 1000000 * sigmann))
#
#     if verbose:
#         print('Cross-section rescaling: ' + str(PbPb_xsec * pp_evtc / (1000000 * sigmann * PbPb_evtc)))
#
#     # R_AA calulation
#     raa = PbPb_jet / pp_jet
#     x = np.asarray(raa.xVals())
#     y = np.asarray(raa.yVals())
#     yerr = np.asarray((raa.yMaxs() - raa.yMins()) / 2)
#     xerr = np.asarray((raa.xMaxs() - raa.xMins()) / 2)
#
#     # Propagate the error from the sums of weights
#     # Depending on the number of events, this step has no significant impact
#     Serr_pp = histos_pp['/_EVTCOUNT'].relErr
#     Serr_PbPb = histos_PbPb['/_EVTCOUNT'].relErr
#     raaerr = np.sqrt(y ** 2 * (Serr_pp ** 2  + Serr_PbPb ** 2) + yerr ** 2)
#
#
#     return [x, y, raaerr, xerr]
