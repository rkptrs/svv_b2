import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *
import os
import scipy
import scipy.stats
from Cit_par import rho0,S,Gamma,Lambda,Temp0,p0,g,R,A,e,c,CD0
import CGCalculations as CGC
import xlrd

CNalpha=False
CN_CT=False
elevatortrimcurvea=False
elevatortrimcurvev=False
FeVe=False


CNalpha=True
#CN_CT=True
#elevatortrimcurvea=True
#elevatortrimcurvev=True
#FeVe=True

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

def polyfitter2(x1,y1):
    
    def func(x, a, c):
        return a+c*x**2

    outlierchecks=5
    coeff,cov=curve_fit(func,x1,y1)
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    y=func(x,coeff[0],coeff[1])


    for j in range(outlierchecks):
        ycheck=func(x1,coeff[0],coeff[1])
        i=0
        e=y1-ycheck
        threshold=2
        deleters=np.array([])
        for i in range(len(e)):
            if abs(scipy.stats.zscore(e)[i])>threshold:
                x1=np.delete(x1,i)
                y1=np.delete(y1,i)
                print ('oulier:',i)
                break
    
    coeff,cov=curve_fit(func,x1,y1)
    y=0
    i=0
    step=(max(x1)-min(x1))/2/len(x1)
    x=np.arange(min(x1),max(x1)+step,step)
    y=func(x,coeff[0],coeff[1])
    return x1,y1,x,y,coeff


workbook = xlrd.open_workbook('Post_Flight_Datasheet.xlsx')

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


# fix thrust.exe singularity at 5th line
#hp1[4]=hp1[4]*1.01
#IAS1[4]=IAS1[4]*1.01
#a1[4]=a1[4]*1.01
#FFl1[4]=FFl1[4]*1.01
#FFr1[4]=FFr1[4]*1.01
#Fused1[4]=Fused1[4]*1.01
#TAT1[4]=TAT1[4]*1.01






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
CmTc=-0.0064
nu=1.47*10**-5




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

    #standard thrust
    FFls=FFrs=np.full(len(hp),0.048)
    data=np.stack((hp,M,Tdiff,FFls,FFrs)).T
    np.savetxt('matlab.dat',data,delimiter=' ')

    os.spawnl(0,"thrust(1).exe",'args')

    T = np.loadtxt( 'thrust.dat' )
    D=np.array([])
    for line in T:
        Ds=np.append(D,sum(line))
    CTs=Ds/(0.5*rho0*Ve**2*S)
    Re=Vt*c/nu

    return Ve,Vetilde,CN,CT,CTs,Wmom,rhomom,Tmom,M,Re


Ve1,Vetilde1,CN1,CT1,CTs1,Wmom1,rhomom1,Tmom1,M1,Re1=conversions(hp1,IAS1,a1,FFl1,FFr1,Fused1,TAT1)
Vetrim,Vetildetrim,CNtrim,CTtrim,CTstrim,Wmomtrim,rhomomtrim,Tmomtrim,Mtrim,Retrim=conversions(hptrim,IAStrim,atrim,FFltrim,FFrtrim,Fusedtrim,TATtrim)
Vecg,Vetildecg,CNcg,CTcg,CTscg,Wmomcg,rhomomcg,Tmomcg,Mcg,Recg=conversions(hpcg,IAScg,acg,FFlcg,FFrcg,Fusedcg,TATcg)




deltacg=0.0254*(CGC.calc_xcg(Fusedcg[1]/4.44822,True)[0]-CGC.calc_xcg(Fusedcg[0]/4.44822,False)[0])

Cmdelta=-1/((decg[1]-decg[0])/180*pi)*np.average(CNcg)*deltacg/c

TC=CNtrim
TCs=CTstrim

deeqstar=detrim-CmTc/Cmdelta*(TCs-TC)

width=16
height=width*9/16
plt.figure(figsize=(width,height))
plt.tick_params(axis='both', labelsize=16)
if CNalpha:
    degree=1
    polyfits=polyfitter(a1,CN1,degree)
    plt.plot(polyfits[0],polyfits[1],'o',label='Data points')
    plt.plot(polyfits[2],polyfits[3],label='Linear approximation $C_{L_0}$='+str(round(polyfits[4][0],4))+' [-]'+'  $C_{L_α}$='+str(round(polyfits[4][1],4))+' [$°^{-1}$]')
    plt.plot(np.empty(0),np.empty(0),' ',label='Mach number range: '+str(round(min(M1),2))+', '+str(round(max(M1),2)))
    plt.plot(np.empty(0),np.empty(0),' ',label='Reynolds number range: '+str(round(min(Re1/10**6),1))+'*$10^6$'+', '+str(round(max(Re1/10**6),1))+'*$10^6$')
    plt.title('Lift curve',fontsize=24)
    plt.xlabel('α [°]',fontsize=20)
    plt.ylabel('$C_L$ [-]',fontsize=20)
    plt.legend(loc='upper left',fontsize=18)
    plt.savefig('Plots/CLalpha')
    
    



if CN_CT:
    polyfits=polyfitter2(CN1,CT1)
    plt.plot(polyfits[1],polyfits[0],'o',label='Data points')
    plt.plot(polyfits[3],polyfits[2],label='Polynomial approximation $C_D$= '+str(round(polyfits[4][0],4))+' +'+str(round(polyfits[4][1],4))+'*$C_L^2$'+' [-]')
    plt.plot(np.empty(0),np.empty(0),' ',label='Mach number range: '+str(round(min(M1),2))+', '+str(round(max(M1),2)))
    plt.plot(np.empty(0),np.empty(0),' ',label='Reynolds number range: '+str(round(min(Re1/10**6),1))+'*$10^6$'+', '+str(round(max(Re1/10**6),1))+'*$10^6$')
    plt.title('Drag polar',fontsize=24)
    plt.xlabel('$C_L$ [-]',fontsize=20)
    plt.ylabel('$C_D$ [-]',fontsize=20)
    plt.legend(loc='lower right',fontsize=18)
    plt.savefig('Plots/CLCD')
    
    print('lift drag polar constants: Cd0,1/(Pi*A*e)', polyfits[4])
    print(CD0,1/A/e/pi)
    emeas=1/polyfits[4][1]/pi/A





if elevatortrimcurvea:
    degree=1
    polyfits=polyfitter(atrim,deeqstar,degree)
    plt.plot(polyfits[0],polyfits[1],'o',label='Data points')
    plt.plot(polyfits[2],polyfits[3],label='Linear approximation '+r'$\frac{dδ_e}{dα}$= '+str(round(polyfits[4][1],4))+' [-]')
    plt.gca().invert_yaxis()
    print ('-Cmalpha/Cmdelta=',polyfits[4][1])
    Cmalpha=polyfits[4][1]*-Cmdelta
    print ('Cmalpha=',Cmalpha)
    plt.title('Elevator trim curve',fontsize=24)
    plt.xlabel('α [°]',fontsize=20)
    plt.ylabel('$δ_{e_{eq}}^\star$ [°]',fontsize=20)
    plt.legend(loc='upper left',fontsize=18)
    plt.savefig('Plots/elevtrimalpha')

if elevatortrimcurvev:
    degree=2
    polyfits=polyfitter(Vetildetrim,deeqstar,degree)
    plt.plot(polyfits[0],polyfits[1],'o',label='Data points')
    plt.plot(polyfits[2],polyfits[3],label='Polynomial approximation $δ_{e_{eq}}^\star= $'+str(round(polyfits[4][0],4))+' +'+str(round(polyfits[4][1],4))+'*$\~V_e$ '+str(round(polyfits[4][2],4))+'*$\~V_e^2$'+' [°]')
    plt.gca().invert_yaxis()
    plt.title('Elevator trim curve',fontsize=24)
    plt.xlabel('$\~V_e$ [m/s]',fontsize=20)
    plt.ylabel('$δ_{e_{eq}}^\star$ [°]',fontsize=20)
    plt.legend(loc='lower left',fontsize=18)
    plt.savefig('Plots/elevtrimV')

if FeVe:
    degree=2
    polyfits=polyfitter(Vetildetrim,Fetrim,degree)
    plt.plot(polyfits[0],polyfits[1],'o',label='Data points')
    plt.plot(polyfits[2],polyfits[3],label='Polynomial approximation $F_e^\star$='+str(round(polyfits[4][0],4))+' '+str(round(polyfits[4][1],4))+'*$\~V_e$ +'+str(round(polyfits[4][2],4))+'*$\~V_e^2$'+' [N]')
    plt.gca().invert_yaxis()
    plt.title('Elevator control force curve',fontsize=24)
    plt.xlabel('$\~V_e$ [m/s]',fontsize=20)
    plt.ylabel('$F_e^\star$ [N]',fontsize=20)
    plt.legend(loc='lower left',fontsize=18)
    plt.savefig('Plots/FeVe')



plt.grid()
plt.show()


