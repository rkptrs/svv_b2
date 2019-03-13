load('flightdata.mat');

%Time - Input in [s]
time = transpose(flightdata.time.data);
save('time.dat','time','-ascii');

%Angle of Attach - Input in [°]
aoa = flightdata.vane_AOA.data;
save('aoa.dat','aoa','-ascii');

%Elevator Trim Tab Deflection - Input in [°]
dte = flightdata.elevator_dte.data;
save('dte.dat','dte','-ascii');

%Elevator Stick Force - Input in [N]
fe = flightdata.column_fe.data;
save('fe.dat','fe','-ascii');

%Left Hand Engine Mass Fuel Flow - Input in [lbs/hr], only output in [kg/s]
lhfmf = (flightdata.lh_engine_FMF.data) * 1.25998e-4;
save('lhfmf.dat','lhfmf','-ascii');

%Right Hand Engine Mass Fuel Flow - Input in [lbs/hr], only output in [kg/s]
rhfmf = (flightdata.rh_engine_FMF.data) * 1.25998e-4;
save('rhfmf.dat','rhfmf','-ascii');

%Total Fuel Used - Input in [lbs], also output in [kg]
lhfulbs = flightdata.lh_engine_FU.data;
rhfulbs = flightdata.rh_engine_FU.data;

tfulbs = lhfulbs + rhfulbs;
tfukg =  tfulbs * 0.453592;

save('tfulbs.dat','tfulbs','-ascii');
save('tfukg.dat','tfukg','-ascii');

%Aileron Deflection - Input in [°]
deltaa = flightdata.delta_a.data;
save('deltaa.dat','deltaa','-ascii');

%Elevator Deflection - Input in [°]
deltae = flightdata.delta_e.data;
save('deltae.dat','deltae','-ascii');

%Rudder Deflection - Input in [°]
deltar = flightdata.delta_r.data;
save('deltar.dat','deltar','-ascii');

%Roll Angle - Input in [°]
rollangle = flightdata.Ahrs1_Roll.data;
save('rollangle.dat','rollangle','-ascii');

%Pitch Angle - Input in [°]
pitchangle = flightdata.Ahrs1_Pitch.data;
save('pitchangle.dat','pitchangle','-ascii');

%Roll Rate - Input in [°/s]
rollrate = flightdata.Ahrs1_bRollRate.data;
save('rollrate.dat','rollrate','-ascii');

%Pitch Rate - Input in [°/s]
pitchrate = flightdata.Ahrs1_bPitchRate.data;
save('pitchrate.dat','pitchrate','-ascii');

%Yaw Rate - Input in [°/s]
yawrate = flightdata.Ahrs1_bYawRate.data;
save('yawrate.dat','yawrate','-ascii');

%Static Air Temperature  - Input in [°C], only output in [K]
tt = (flightdata.Dadc1_tat.data) + 273;
save('tt.dat','tt','-ascii');

%Pressure Altitude - Input in [ft], only output in [m]
hp = (flightdata.Dadc1_alt.data) * 0.3048;
save('hp.dat','hp','-ascii');

%Mach number [-]
mach = flightdata.Dadc1_mach.data;
save('mach.dat','mach','-ascii');

%True Airspeed - Input in [kts], only output in [m/s]
tas = (flightdata.Dadc1_tas.data) * 0.514444;
save('tas.dat','tas','-ascii');

%Measurement Running [True/False]
measrunning = flightdata.measurement_running.data;
save('measrunning.dat','measrunning','-ascii');

%Number of Measurement Ready [n]
nomeasrdy = flightdata.measurement_n_rdy.data;
save('nomeasrdy.dat','nomeasrdy','-ascii');