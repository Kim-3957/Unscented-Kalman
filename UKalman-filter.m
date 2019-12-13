function [result]=Ukalman_vel_HRC(Mdata,Sdata,Svel,dt,Q,R);
Pos =      Sdata'; 
Vel =       Svel';
True =     Mdata';
XEst = [Pos(1) Vel(1)]'; %% [x v]'
PEst = eye(2);           %% Initial states
time = dt(1,1);
result.XPred=[XEst'];
result.time=[time];
result.XEst=[XEst'];
result.XMes=[XEst'];
result.PEst=[diag(PEst)'];
H = [1 0];
% UKF Parameter
alpha=0.001;
beta =2;
kappa=0;

n=length(XEst);%size of state vector
lamda=alpha^2*(n+kappa)-n;

%calculate weights
wm=[lamda/(lamda+n)];
wc=[(lamda/(lamda+n))+(1-alpha^2+beta)];
for i=1:2*n
    wm=[wm 1/(2*(n+lamda))];
    wc=[wc 1/(2*(n+lamda))];
end
gamma=sqrt(n+lamda);

% Main loop
for i =2:length(dt)
    time = time + dt(i); 
    F = [1 dt(i);
         0 1 ];
    % ------ Unscented Kalman Filter --------
    % X Sigma Predict 
    sigma=GenerateSigmaPoints(XEst,PEst,gamma);
    sigma=PredictMotion(sigma,F);
    XPred=(wm*sigma')';
    % P Predict
    PPred=CalcSimgaPointsCovariance(XPred,sigma,wc,Q);
    % Update
    XMes = [Pos(i) Vel(1)]'; % An actual measurement 
    y = XMes - H*XPred ;       % Comparion between measurement and predicted value
    sigma=GenerateSigmaPoints(XPred,PPred,gamma);
    zSigma=PredictObservation(sigma,H);
    zb=(wm*sigma')';
    St=CalcSimgaPointsCovariance(zb,zSigma,wc,R);
    Pxz=CalcPxz(sigma,XPred,zSigma,zb,wc);
    K=Pxz*inv(St);
    XEst = XPred + K*y;
    PEst=PPred-K*St*K';
    
    % Simulation Result
    result.XPred=[result.XPred;XPred'];
    result.time=[result.time; time];
    result.XEst=[result.XEst; XEst'];
    result.XMes=[result.XMes; XMes'];
    result.PEst=[result.PEst; diag(PEst)'];   
    
end
end
