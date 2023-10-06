clear

%% XOR
clear

NN = MLP;
NN.bias = true;
NN.normalize = true;
NN.neuronNum = [2 2 1];
NN.learningRate = 0.01;
NN.Kmax = 10000;
NN.Emax = 0.001;
NN.acctivationFunction = "reLU";

data = [0 0 0; 1 0 1; 0 1 1; 1 1 0]';

X = data(1: end - 1, :);
Y = data(end, :);

NN.X = X;
NN.Y = Y;

NN = NN.set_initial_random_val;
NN = NN.train(X);
NN = NN.calculate_output(X);

disp(NN)

%% Cal_House
clear 

load('cal_housing.mat', 'data');

NN = MLP;
NN.bias = true;
NN.normalize = true;
NN.neuronNum = [8 32 32 8 1];
NN.learningRate = 0.01;
NN.Kmax = 200;
NN.Emax = 0.000000001;
NN.acctivationFunction = "reLU";

data = data(:, randperm(size(data, 2)));

if NN.normalize 
   Xmax = max(data, [], 2);
   Xmin = min(data, [], 2);

   data = (data - Xmin) ./ (Xmax - Xmin);
end

X = data(1: end - 1, :);
Y = data(end, :);

X_train = X(:, 1: 0.7*size(data, 2)-1);
Y_train = Y(1: 0.7*size(data, 2)-1);

X_test = X(:, 0.7*size(data, 2): end);
Y_test = Y(:, 0.7*size(data, 2): end);

NN.X = X_train;
NN.Y = Y_train;

NN = NN.set_initial_random_val;
NN = NN.train(X_train);
NN = NN.test(X, Y);

disp(NN)