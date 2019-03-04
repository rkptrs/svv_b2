# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 15:28:12 2019
@author: jonas
"""
import numpy as np

def inpounds(massinkg): #converts kg to pounds [lbs], should work for both single numbers and arrays
    massinlbs = massinkg * 1/0.453592
    return massinlbs

def currentfuelmoment(fuelinlbs): #outputs fuel moment based on input of fuel weight in lbs
    xfuelinlbs = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5008])
    yfuelmoment = np.array([298.16, 591.18, 879.08, 1165.42, 1448.40, 1732.53, 2014.80, 2298.84, 2581.92, 2866.30, 3150.18, 3434.52, 3718.52, 4003.23, 4287.76, 4572.24, 4856.56, 5141.16, 5425.64, 5709.90, 5994.04, 6278.47, 6562.82, 6846.96, 7131.00, 7415.33, 7699.60, 7984.34, 8269.06, 8554.05, 8839.04, 9124.80, 9410.62, 9696.97, 9983.40, 10270.08, 10556.84, 10843.87, 11131.00, 11481.20, 11705.50, 11993.31, 12281.18, 12569.04, 12856.86, 13144.73, 13432.48, 13720.56, 14008.46, 14320.34])

    z1 = np.polyfit(xfuelinlbs, yfuelmoment, 3)
    fuelmomentfunction = np.poly1d(z1)

    currentfuelmoment = fuelmomentfunction(fuelinlbs)
    return currentfuelmoment
    
def calc_payload(masses, xpositions): #calculates total payload weight
    sumpayloadmass = np.sum(masses) #sum of all weights
    sumpayloadmoment = np.sum(masses * xpositions)
    
    return sumpayloadmass, sumpayloadmoment
    
payloadmassesinkg = np.array([95, 92, 83, 76, 51, 77, 77, 84, 91, 0, 0, 0]) #In order of Pilot 1, Pilot 2, Coordinator, 1L, 1R, 2L, 2R, 3L, 3R, nosebaggage, aftbaggage1, aftbaggage2
payloadmassesinlbs = inpounds(payloadmassesinkg)

xpositionsininch = np.array([131, 131, 170, 214, 214, 251, 251, 288, 288, 74, 321, 338])

fuelloadinlbs = 4000.