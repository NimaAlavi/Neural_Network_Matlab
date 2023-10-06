classdef MLP 
   properties
      bias 
      normalize
      learningRate
      Kmax
      Emax
      neuronNum
      acctivationFunction
      meanWeight = 0;
      std = 0.1;
      X
      Y
      W
      E
   end
   methods
      function obj = set_initial_random_val(obj)
         if obj.bias 
            if obj.normalize
               for i = 1: length(obj.neuronNum) - 1
                  obj.W(i).val = normrnd(obj.meanWeight, obj.std, [obj.neuronNum(i+1), obj.neuronNum(i) + 1]);
               end
            else
               for i = 1: length(obj.neuronNum) - 1
                  obj.W(i).val = randn(obj.neuronNum(i+1), obj.neuronNum(i) + 1);
               end
            end
         else
            if obj.normalize
               for i = 1: length(obj.neuronNum) - 1
                  obj.W(i).val = normrnd(obj.meanWeight, obj.std, [obj.neuronNum(i+1), obj.neuronNum(i)]);
               end
            else
               for i = 1: length(obj.neuronNum) - 1
                  obj.W(i).val = randn(obj.neuronNum(i+1), obj.neuronNum(i));
               end
            end
         end       
      end

      function dataOut = acctivationFunc(x, obj)
         if obj.acctivationFunction == "sigmoid_bipolar"
            dataOut = 2./(1 + exp(-x)) - 1;
         elseif obj.acctivationFunction == "sigmoid_unipolar"
            dataOut = 1./(1 + exp(-x));
         elseif obj.acctivationFunction == "reLU"
            dataOut = max(x, 0);
         elseif obj.acctivationFunction == "tanh"
            dataOut = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
         end
      end
      function dataOut = acctivationFunc_dot(x, obj)
         if obj.acctivationFunction == "sigmoid_bipolar"
            dataOut = 1/2./(1 - x .^ x);
         elseif obj.acctivationFunction == "sigmoid_unipolar"
            dataOut = x .* (1-x);
         elseif obj.acctivationFunction == "reLU"
            dataOut = max(x, 0);
            dataOut(dataOut == 0) = 0;
            dataOut(dataOut ~= 0) = 1;
         elseif obj.acctivationFunction == "tanh"
            dataOut = 1 - tanhyper(x) .^2;
         end
      end
      
      function dataOut = calculate_output(obj, X)
         if obj.bias 
            X = [X; ones(1, length(X(1, :)))];
         end
         for i = 1: length(obj.neuronNum) - 1
            X = acctivationFunc(obj.W(i).val * X, obj);
            if obj.bias && i < length(obj.neuronNum) - 1
               X = [X; ones(1, length(X(1, :)))];
            end
         end
      
         dataOut = X;
      end
      
      function obj = update_weight(obj)
         layers(:).val = zeros(2, 1);
         for p = 1: size(obj.X, 2)
            obj.E = 0;
            layers(1).val = obj.X(:, p);
            if obj.bias 
               layers(1).val = [layers(1).val; 1];
            end
            for i = 2: length(obj.neuronNum)
               layers(i).val = acctivationFunc(obj.W(i-1).val * layers(i-1).val, obj);
               if obj.bias && i < length(obj.neuronNum)
                  layers(i).val = [layers(i).val; 1];
               end
            end
            obj.E = obj.E + 1/2 * (obj.Y(p)-layers(end).val).^2;
            Wcopy = obj.W;
            deltha(length(layers)).val = (obj.Y(p) - layers(end).val) .* acctivationFunc_dot(Wcopy(end).val * layers(end-1).val , obj);
            for i = length(layers)-1: -1: 2
               deltha_p = (Wcopy(i).val' * deltha(i+1).val);
               if obj.bias
                  deltha_p = deltha_p(1: end-1);
               end
               deltha(i).val = deltha_p .* acctivationFunc_dot(Wcopy(i-1).val * layers(i-1).val, obj);
            end
      
            for i = 1: length(layers)-1
               obj.W(i).val = obj.W(i).val + obj.learningRate * (deltha(i+1).val * layers(i).val');
            end
         end
      end
      
      function obj = train(obj, X)
         reported_E = zeros(obj.Kmax, 1);
         for k = 1: obj.Kmax
            obj = update_weight(obj);
            O = calculate_output(obj, X);
            MSE = mean((obj.Y - O).^2);
            reported_E(k) = MSE;
            fprintf('%d th training cycle ====> MSE Error: %f \n', k, MSE);
            if obj.E < obj.Emax
               break;
            end
         end
         plot(reported_E)
      end
      
      function MSE = test(obj, X, Y)
         O = calculate_output(obj, X);
         MSE = mean((Y - O).^2);
         fprintf('testing MSE is: %f', MSE);
      end

   end
end