load('breast_cancer.mat');

%% Process withOut Bias
X = data(:, 1: end - 1);
Y = data(:, end);

X_train = X(1: 450, :);
Y_train = Y(1: 450, :);
X_test = X(451: end, :);
Y_test = Y(451: end, :);

activationFunc = "reLU";            % "sigmoid", "reLU", "tanh"
learning_rate = 0.001;
normalize = false;

w = set_initial_random_weight(X, 0, 0.1, normalize);
w = train(X_train, Y_train, w, 1, 300, normalize, learning_rate, activationFunc);
test(X_test, Y_test, w, true, activationFunc);

%% Process with Bias 1
X = [data(:, 1: end - 1) ones(length(data), 1)];
Y = data(:, end);

X_train = X(1: 450, :);
Y_train = Y(1: 450, :);
X_test = X(451: end, :);
Y_test = Y(451: end, :);

activationFunc = "reLU";            % "sigmoid", "reLU", "tanh"
learning_rate = 0.01;
normalize = true;

w = set_initial_random_weight(X, 0, 0.1, normalize);
w = train(X_train, Y_train, w, 10, 300, normalize, learning_rate, activationFunc);
test(X_test, Y_test, w, true, activationFunc);

%% Functions
function w1 = set_initial_random_weight(x, mean, std, normal)
   dataNum = size(x);
   if normal
      w1 = normrnd(mean, std, [1, dataNum(2)]);
   else
      w1 = randn(1, dataNum(2));
   end
end

function dataOut = sigmoid(x)
   dataOut = 1./(1 + exp(-x));
end

function dataOut = reLU(x)
   dataOut = max(x, 0);
end

function o = calculate_output(x, w, activationFunc)
   if activationFunc == "sigmoid"
      o = sigmoid(sum(x .* w, 2));
   elseif activationFunc == "reLU"
      o = reLU(sum(x .* w, 2));
   else 
      o = tanh(sum(x .* w, 2));
   end
end

function [E, w] = update_weight(x, y, w, learning_rate, activationFunc)
   E = 0;
   dataNum = size(x);
   for p = 1: dataNum(1)
      o = calculate_output(x(p, :), w, activationFunc);
      E = E + 1/2 * (y(p) - o).^2;
      w = w + 1/2 * learning_rate * (y(p) - o) .* (1 - o.^2) .* x(p, :);
   end
end

function w = train(x, y, w, Emax, Kmax, normalize, learning_rate, activationFunc)
   if normalize
      Xi_max = max(x);
      Xi_min = min(x);
      if Xi_max(end) == Xi_min(end)
         Xi_min(end) = 1;
         Xi_max(end) = 0;
      end
      x = (x - Xi_min) ./ (Xi_max - Xi_min);
   end
   reported_E = zeros(Kmax, 1);
   dataNum = size(x);
   for k = 1: Kmax
      [E, w] = update_weight(x, y, w, learning_rate, activationFunc); 
      MSE = mean((y - calculate_output(x, repmat(w, dataNum(1), 1), activationFunc)).^2);
      reported_E(k) = MSE;
      fprintf('%d th training cycle ====> MSE Error: %f \n', k, MSE);
      if E < Emax
         break;
      end
   end
   plot(reported_E)
end

function MSE = test(x, y, w, normalize, activationFunc)
   if normalize
      Xi_max = max(x);
      Xi_min = min(x);
      if Xi_max(end) == Xi_min(end)
         Xi_min(end) = 1;
         Xi_max(end) = 0;
      end
      x = (x - Xi_min) ./ (Xi_max - Xi_min);
   end
   dataNum = size(x);
   MSE = mean((y - calculate_output(x, repmat(w, dataNum(1), 1), activationFunc)).^2);
   fprintf('testing MSE is: %f', MSE);
end

