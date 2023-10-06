clear;

generics.zetha = 0.01;
generics.layersNum = [1 2 1];
generics.clusters = generics.layersNum(2);
generics.bias = randn(1, generics.layersNum(3));
generics.Emax = 0.001;
generics.kmax = 100;

X = sort(unifrnd(0, 1, [100, 1]));
noise = unifrnd(-0.1, 0.1, [100, 1]);
Y = sin(2*pi*X) + noise;

generics.Y = sin(2*pi*X) + noise;
generics = train(generics, X);

F = predict(generics, X);

plot(F)
hold all
plot(Y)

function [centroid, sigma] = doKmean(generics, X)
   KM          = K_means;
   KM.dataIn   = X;
   KM.clusters = generics.clusters;
   KM          = KM.setCenterInitial;
   KM          = KM.doClustring;
   centroid    = KM.centroid;
   
   sigma = 0;
   for i = 1: length(centroid)
      for j = i: length(centroid)
         dis = norm(centroid(i) - centroid(j));
         if dis >= sigma
            sigma = dis;
         end
      end
   end
   sigma = sigma / sqrt(length(centroid));
end

function W = set_initial_weigth(generics)
   layersNum = generics.layersNum;
   W = randn(layersNum(2), layersNum(3));
end

function f = GausianActvFunc(centroid, sigma, x)
   f = zeros(size(x, 1), 1);
   for i = 1: size(x, 1)
      f(i) = exp(-norm(x(i) - centroid)^2/sigma^2);
   end
end

function H = calcHidden(generics, X)
   H = zeros(size(X, 1), generics.layersNum(2));
   for i = 1: size(H, 2)
      H(:, i) = GausianActvFunc(generics.centroid(i), generics.sigma, sum(X, 2));
   end
end

function generics = train(generics, X)
   [centroid, sigma] = doKmean(generics, X);
   generics.centroid = centroid;
   generics.sigma    = sigma;
   generics.W        = set_initial_weigth(generics);
   H                 = calcHidden(generics, X);
   for k = 1: generics.kmax
      generics.Error = 0;
      for p = 1: size(X, 1)
         O              = H(p, :) * generics.W + generics.bias;
         generics.Error = generics.Error + 1/2 * (O - generics.Y(p, :)).^2;
         generics.W     = generics.W - generics.zetha * H(p, :)' * (O - generics.Y(p, :));
         generics.bias  = generics.bias - generics.zetha * (O - generics.Y(p, :));
      end
      if generics.Error < generics.Emax
         break
      end
   end
end

function predict = predict(generics, X)
   predict = calcHidden(generics, X) * generics.W + generics.bias;
end







