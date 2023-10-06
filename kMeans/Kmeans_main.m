clear 

%% 
load('dataimg.mat')

KM = Kmeans;
KM.dataIn = data(1)';
KM.clusters = 64;
KM = KM.setCenterInitial;
KM = KM.doClustring;
label = KM.label;
centroid = KM.centroid;







