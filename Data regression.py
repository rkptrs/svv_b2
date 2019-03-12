import numpy as np
import matplotlib.pyplot as plt
from math import *
import os
import scipy
import scipy.stats
from Cit_par import rho0,S,Gamma,Lambda,Temp0,p0,g,R,A,e,c
import CGCalculations as CGC
import xlrd

CNalpha=False
CN_CT=False
elevatortrimcurve=True

def polyfitter(x1,y1,degree):
    outlierchecks=5
    coefficients=np.flip(np.polyfit(x1,y1,degree))
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    for cff in coefficients:
        y+=cff*x**i
        i+=1


    for j in range(outlierchecks):
        ycheck=0
        i=0
        for cff in coefficients:
            ycheck+=cff*x1**i
            i+=1
        e=y1-ycheck
        threshold=2
        deleters=np.array([])
        for i in range(len(e)):
            if abs(scipy.stats.zscore(e)[i])>threshold:
                x1=np.delete(x1,i)
                y1=np.delete(y1,i)
                print ('oulier:',i)
                break
    
    coefficients=np.flip(np.polyfit(x1,y1,degree))
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    for cff in coefficients:
        y+=cff*x**i
        i+=1
    return x1,y1,x,y,coefficients


workbook = xlrd.open_workbook('REFERENCE_Post_Flight_Datasheet_Flight.xlsx')

sheet = workbook.sheet_by_name('Sheet1')


hp1=np.array([])
IAS1=np.array([])
a1=np.array([])
FFl1=np.array([])
FFr1=np.array([])
Fused1=np.array([])
TAT1=np.array([])
for i in range(6):####change to 7 for real data
    hp1=np.append(hp1,float(sheet.col(3)[27+i].value)*0.3048)           #m
    IAS1=np.append(IAS1,float(sheet.col(4)[27+i].value)*0.51444)        #m/s
    a1=np.append(a1,float(sheet.col(5)[27+i].value))                    #degrees
    FFl1=np.append(FFl1,float(sheet.col(6)[27+i].value)*0.000125998)    #kg/s
    FFr1=np.append(FFr1,float(sheet.col(7)[27+i].value)*0.000125998)    #kg/s
    Fused1=np.append(Fused1,float(sheet.col(8)[27+i].value)*4.44822)    #Newtons
    TAT1=np.append(TAT1,float(sheet.col(9)[27+i].value)+273.15)         #Kelvin

hptrim=np.array([])
IAStrim=np.array([])
atrim=np.array([])
detrim=np.array([])
detrtrim=np.array([])
Fetrim=np.array([])
FFltrim=np.array([])
FFrtrim=np.array([])
Fusedtrim=np.array([])
TATtrim=np.array([])
for i in range(7):
    hptrim=np.append(hptrim,float(sheet.col(3)[58+i].value)*0.3048)         #m
    IAStrim=np.append(IAStrim,float(sheet.col(4)[58+i].value)*0.51444)      #m/s
    atrim=np.append(atrim,float(sheet.col(5)[58+i].value))                  #degrees
    detrim=np.append(detrim,float(sheet.col(6)[58+i].value))                #degrees
    detrtrim=np.append(detrtrim,float(sheet.col(7)[58+i].value))            #degrees
    Fetrim=np.append(Fetrim,float(sheet.col(8)[58+i].value))                #Newtons  
    FFltrim=np.append(FFltrim,float(sheet.col(9)[58+i].value)*0.000125998)  #kg/s
    FFrtrim=np.append(FFrtrim,float(sheet.col(10)[58+i].value)*0.000125998) #kg/s
    Fusedtrim=np.append(Fusedtrim,float(sheet.col(11)[58+i].value)*4.44822) #Newtons
    TATtrim=np.append(TATtrim,float(sheet.col(12)[58+i].value)+273.15)      #Kelvin


hpcg=np.array([])
IAScg=np.array([])
acg=np.array([])
decg=np.array([])
detrcg=np.array([])
Fecg=np.array([])
FFlcg=np.array([])
FFrcg=np.array([])
Fusedcg=np.array([])
TATcg=np.array([])
for i in range(2):
    hpcg=np.append(hpcg,float(sheet.col(3)[74+i].value)*0.3048)         #m
    IAScg=np.append(IAScg,float(sheet.col(4)[74+i].value)*0.51444)      #m/s
    acg=np.append(acg,float(sheet.col(5)[74+i].value))                  #degrees
    decg=np.append(decg,float(sheet.col(6)[74+i].value))                #degrees
    detrcg=np.append(detrcg,float(sheet.col(7)[74+i].value))            #degrees
    Fecg=np.append(Fecg,float(sheet.col(8)[74+i].value))                #Newtons  
    FFlcg=np.append(FFlcg,float(sheet.col(9)[74+i].value)*0.000125998)  #kg/s
    FFrcg=np.append(FFrcg,float(sheet.col(10)[74+i].value)*0.000125998) #kg/s
    Fusedcg=np.append(Fusedcg,float(sheet.col(11)[74+i].value)*4.44822) #Newtons
    TATcg=np.append(TATcg,float(sheet.col(12)[74+i].value)+273.15)      #Kelvin


Winit=14064.765428 #lbs
Winit=4.44822*Winit # newtons
Ws=60500 # newtons





def conversions(hp,V,alpha,FFl,FFr,Wf,TAT):
    Wmom=Winit-Wf

    pmom=p0*(1+Lambda*hp/Temp0)**(-g/Lambda/R)

    M=(2/(Gamma-1)*((1+p0/pmom*((1+(Gamma-1)/2/Gamma*rho0/p0*V**2)**(Gamma/(Gamma-1))-1))**((Gamma-1)/Gamma)-1))**0.5
    Tmom=TAT/(1+(Gamma-1)/2*M**2)
    TmomISA=Temp0+Lambda*hp

    Tdiff=Tmom-TmomISA
    Vt=(Gamma*R*TmomISA)**0.5*M
    rhomom=pmom/R/TmomISA
    Ve=Vt*(rhomom/rho0)**0.5
    Vetilde=Ve*(Ws/Wmom)**0.5
    
    CN=Wmom/(0.5*rho0*Ve**2*S)
    data=np.stack((hp,M,Tdiff,FFl,FFr)).T
    np.savetxt('matlab.dat',data,delimiter=' ')


    os.spawnl(0,"thrust(1).exe",'args')

    T = np.loadtxt( 'thrust.dat' )
    D=np.array([])
    for line in T:
        D=np.append(D,sum(line))


    CT=D/(0.5*rho0*Ve**2*S)


    return Ve,CN,CT,Wmom,rhomom,Tmom

#Ve1,CN1,CT1,Wmom1,rhomom1,Tmom1=conversions(hp1,IAS1,a1,FFl1,FFr1,Fused1,TAT1)
Vetrim,CNtrim,CTtrim,Wmomtrim,rhomomtrim,Tmomtrim=conversions(hptrim,IAStrim,atrim,FFltrim,FFrtrim,Fusedtrim,TATtrim)
Vecg,CNcg,CTcg,Wmomcg,rhomomcg,Tmomcg=conversions(hpcg,IAScg,acg,FFlcg,FFrcg,Fusedcg,TATcg)

#deeqstar=de-CmTc/Cmdelta(TCs-Tc)

deltacg=0.0254*(CGC.calc_xcg(Fusedcg[1]/4.44822,True)[0]-CGC.calc_xcg(Fusedcg[0]/4.44822,False)[0])

Cmdelta=-(decg[1]-decg[0])*180/pi*np.average(CNcg)*deltacg/c


if CNalpha:
    degree=1
    polyfits=polyfitter(a1,CN1,degree)
    plt.plot(polyfits[0],polyfits[1],'o')
    plt.plot(polyfits[2],polyfits[3])

    print(polyfits[4])



if CN_CT:
    degree=2
    polyfits=polyfitter(CN1,CT1,degree)
    plt.plot(polyfits[1],polyfits[0],'o')
    plt.plot(polyfits[3],polyfits[2])

    print('lift drag polar constants: Cd0,k,1/(Pi*A*e)', polyfits[4])
    print(1/A/e/pi)

if elevatortrimcurve:
    degree=1
    polyfits=polyfitter(atrim,detrim,degree)
    plt.plot(polyfits[0],polyfits[1],'o')
    plt.plot(polyfits[2],polyfits[3])
    plt.gca().invert_yaxis()
    print ('-Cmalpha/Cmdelta=',polyfits[4][1])
    Cmalpha=polyfits[4][1]*-Cmdelta
    print ('Cmalpha=',Cmalpha)
plt.show()


