import math
import numpy as np





def GMST(yy, mm, dd, hh, min, ss):
    OMEGA_E = 7.2921150 * 10 ** (-5)
    JD_2000 = 2451545.0
    CENTURY = 36525
    c = (julian_date(yy, mm, dd, 0, 0, 0) - JD_2000) / CENTURY
    t = hh*3600 + min*60 + ss
    gmst = ((-6.2E-6*c + 0.093104)*c + 8640184.812866)*c + 24110.54841
    gmst = gmst*(math.pi/43200) + OMEGA_E*t % 2*math.pi
    return gmst

def julian_date(yy, mm, dd, hh, min, ss):
    m = np.fix((mm-14)/12)
    jd = np.fix(dd-32075+(1461*(yy+4800+m))/4) + np.fix(367*(mm-2-12*m)/12) - np.fix(3*np.fix((yy+4900+m)/100)/4)
    jd = jd+(hh-12)/24+min/1440+ss/86400
    return jd

# -*- coding: utf-8 -*-

