function [X,P]=Ukalman_vel_realtime(X,P,Sdata,Svel,dt,Q,R);
%%%Initialization%%%
F = [1 dt;0 1 ];    % Constant Velocity model  
H = [1 0];          % Observation matrix
%%%UKF Parameter%%%
alpha=0.001;        % Parameter for the UKF filter
beta =2;            % Parameter for the UKF filter
kappa=0;            % parameter for the UKF filter
n=length(X);        %size of state vector
lamda=alpha^2*(n+kappa)-n;          % Lamda for the sigma points
%%%Calculate weights of the sigma points%%%
wm=[lamda/(lamda+n)];
wc=[(lamda/(lamda+n))+(1-alpha^2+beta)];
for i=1:2*n
    wm=[wm 1/(2*(n+lamda))];
    wc=[wc 1/(2*(n+lamda))];
end
gamma=sqrt(n+lamda);
%%% ---Prediction stage--- %%%
%%% Priori Sigma Predict %%%
sigma=GenerateSigmaPoints(XEst,PEst,gamma); % Cautious! sigma is a point not a variation
sigma=PredictMotion(sigma,F);               % Use state estimate matrix for all of the sigma points
%%% Priori X Sigma Predict %%%
XPred=(wm*sigma')';                         % Expectation of sigma points are assumed as XPred 
%%% Priori P Sigma Predict %%%
PPred=CalcSimgaPointsCovariance(XPred,sigma,wc,Q);
%%% ---Measurement stage--- %%%
z = Sdata;                                  % An actual measurement 
y = z - H*XPred ;                           % Comparion between measurement and predicted value
%%% ---Update stage--- %%%
%%% Post Sigma estimate %%%
sigma=GenerateSigmaPoints(XPred,PPred,gamma);
zSigma=PredictObservation(sigma,H);
%%% Post X Sigma estimate %%%
zb=(wm*sigma')';
%%% Priori P Sigma estimate %%%
St=CalcSimgaPointsCovariance(zb,zSigma,wc,R);
%%% Kalman gain estimate %%%
Pxz=CalcPxz(sigma,XPred,zSigma,zb,wc);
K=Pxz*inv(St);
%%% Update %%%
XEst = XPred + K*y;
PEst=PPred-K*St*K';
end
