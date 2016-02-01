'''

'''

import numpy as np

# Upper-case X,Y: Training examples scaled to actual range of inputs & outputs
# Lower-case x,y: Training examples scaled to network inputs & outputs

class NeuralNet:
    def __init__(self):
        # Neural net settings defined by user
        self.output_type = ['logistic', 'linear']
        self.learning_rate = 0.1
        self.epochs = 10
        self.current_epoch = 0
        self.disp = 1
        self.reg_lambda = 0 # Parameter for regularization
        # Performance metrics
        self.J_hist = []
        self.accuracy_hist = []
        # Network topology
        self.input_range = []
        self.inputs = None
        self.outputs = None
        self.layer_size = []
        # Thetas variables
        self.Thetas = []
        self.Thetas_ur = [] # Unrolled vector of Theta parameters
        # Inputs-outputs transformation, computed when training data is set
        self.X = np.array([])
        self.Y = np.array([])
        self.Xmean = np.array([])
        self.Xstd = np.array([])
        self.Ymean = np.array([]) # Only used for linear_regression type
        self.Ystd = np.array([])
        self.training_examples = None
        # Back-propagation variables
        self.activations = [] # Layers activations
        self.layer_inputs = [] # Inputs to the layer, before activating the neuron
        self.epsilon_l = []
        self.delta_l = []
    
    # Receives an un-rolled vector of Thetas and returns a tuple with the cost 
    # and gradients 
    def __call__(self, Thetas_ur):
        self.current_epoch += 1
        print("Epoch = ", self.current_epoch) 
        self.reshapeThetas(Thetas_ur) # Reshapes the thetas and sets it to the structure
        self.predict()
        self.gradients() # Compute gradients
        J = self.cost()
        grad = self._unroll(self.delta_l)
        if self.output_type == 'logistic':
            accuracy = self.accuracy()
            J = 100.0 - accuracy
        print("Cost: ", J) 
        if self.output_type == 'logistic': print('Accuracy: ', accuracy) 
        print()
        #if self.current_epoch >= self.epochs: J = 0
        return J, np.array(grad)
    
    def train(self, epochs=None, learning_rate=None, disp=None):
        if epochs == None: epochs = self.epochs
        if learning_rate == None: learning_rate = self.learning_rate
        if disp == None: disp = self.disp
        self.accuracy_hist = []
        self.J_hist = []
        for epoch in range(epochs):
            self.predict()
            self.gradients() # Compute gradients
            self.updateWeights(learning_rate)
            J = self.cost()
            self.J_hist.append(J)
            if self.output_type == 'logistic':
                accuracy = self.accuracy() 
                self.accuracy_hist.append(accuracy)
            if (epoch+1)%disp == 0: 
                print("Epoch: ", epoch+1, " / ", epochs)
                print("Cost: ", J) 
                if self.output_type == 'logistic': print('Accuracy: ', accuracy)
                print()
        return self.J_hist
    
    def adjustShape(self, V, cols):
        '''
        Adjust the shape of training examples or inputs to the network.
        '''
        if len(V.shape) == 1:
            # One column, many training examples
            if cols == 1: return np.reshape(V, (V.size, 1))
            # One training example, many columns
            elif cols > 1: return np.reshape(V, (1, V.size))
        return V
    
    def setTrainingData(self, X, Y, epochs=None, output_type=None, 
                        learning_rate=None, disp=None, reg_lambda=None):
        '''
        Read inputs-outputs to determine range transformations
        X -> (X - X.mean(0))/X.std(0)
        Y -> [0,1]    
        '''
        self.X = X
        self.Y = Y
        self.X = self.adjustShape(self.X, self.inputs)
        self.Y = self.adjustShape(self.Y, self.outputs)
        self.Xmean = self.X.mean(0)
        self.Xstd = self.X.std(0)
        self.Xstd[self.Xstd < 1.0] = 1.0 # Avoid inf 
        if self.output_type == 'linear':
            self.Ymean = self.Y.mean(0)
            self.Ystd = self.Y.std(0)
            self.Ystd[self.Ystd < 1.0] = 1.0 # Avoid inf
        else: # Logistic regression mean=0, std=1, thus Y is not transformed
            self.Ymean = np.zeros((1,self.Y.shape[1]))
            self.Ystd = np.ones((1,self.Y.shape[1]))
        if epochs != None: self.epochs = epochs
        if output_type != None: self.output_type = output_type
        if learning_rate != None: self.learning_rate = learning_rate
        if disp != None: self.disp = disp
        if reg_lambda != None: self.reg_lambda = reg_lambda
        self.training_examples = self.X.shape[0]
            
    # ~~~~~ Thetas operations ~~~~~     
    
    # Initialize Thetas in the range [0,1]
    def initializeThetas(self):
        self.Thetas = []
        layers_size = [self.inputs] + self.layer_size + [self.outputs]
        for i in range(len(layers_size)-1):
            nrows, ncols = layers_size[i+1], layers_size[i]+1 # Add bias term
            self.Thetas.append(np.random.rand(nrows, ncols))
    
    # Transform the Theta's matrices into a vector with all their weights
    def unrollThetas(self, Thetas=None):
        if Thetas == None: Thetas = self.Thetas
        self.Thetas_ur = []
        for Theta in Thetas: # ~~~ CHANGE FOR APPLY!!! ~~~
            self.Thetas_ur += list(np.reshape(Theta, Theta.size))
        return self.Thetas_ur
    
    # Transform the Theta's matrices into a vector with all their weights
    def _unroll(self, matrix_list=None):
        M_ur = []
        for M in matrix_list: M_ur += list(np.reshape(M, M.size))
        return M_ur
    
    
    #' Params:
    #' Theta_ur: Un-rolled vector of Theta parameters
    def reshapeThetas(self, Thetas_ur=None):
        if Thetas_ur == None: Thetas_ur = self.Thetas_ur
        layers_size = [self.inputs] + self.layer_size + [self.outputs]
        i_e = 0
        Thetas = []
        for i in range(len(layers_size)-1):
            nrows, ncols = layers_size[i+1], layers_size[i]+1 # Add bias term
            i_s, i_e = i_e, i_e + (nrows * ncols) # Start & end indices
            Thetas.append(np.reshape(Thetas_ur[i_s:i_e], (nrows, ncols)))
        self.Thetas = Thetas
        return self.Thetas
    
    # Propagation & Back-propagation functions   
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoidGrad(self, z): return self.sigmoid(z) * (1-self.sigmoid(z))
    
    def addBias(self, x):
        xb = np.ones((x.shape[0], x.shape[1]+1))
        xb[:,1:] = x
        return xb
    
    def predict(self, X=None):
        '''
        Propagate the inputs through the network.
        '''
        if X == None: X = self.X # Use training data if not provided
        x = self.adjustShape(X, self.inputs) # Adjust data to number of inputs
        x = (x - self.Xmean) / self.Xstd # Scale to [-1,1] with computed range
        prev_out = np.array(self.addBias(x))
        # Inputs, X are not transformed by sigmoid, so activations are the same
        self.layer_inputs = [prev_out] 
        self.activations = [prev_out]
        if len(self.Thetas) > 1: # One or more hidden layers
            for Theta in self.Thetas[:-1]: # All weights except output layer
                # Inputs to the layer (neurons only), considering bias
                prev_out = np.dot(prev_out, Theta.T)
                self.layer_inputs.append(prev_out) # Known as z
                # Propagate last output through the next layer and add bias
                prev_out = self.addBias(self.sigmoid(prev_out))
                self.activations.append(prev_out)
        # Output of the network
        prev_out = np.dot(prev_out, self.Thetas[-1].T) 
        self.layer_inputs.append(prev_out) # Net output
        if self.output_type == 'logistic': # Output probabilities instead
            prev_out = self.sigmoid(prev_out)
        self.activations.append(prev_out)
        return prev_out*self.Ystd + self.Ymean # Scale back output to original range
    
    def gradients(self, Y=None):
        if Y == None: Y = self.Y
        y = self.adjustShape(Y, self.outputs)
        y = (y - self.Ymean) / self.Ystd # Scale to network range
        e = self.activations[-1] - y # Output error
        self.epsilon_l = [e]
        self.delta_l = [np.dot(e.T, self.activations[-2])] # Gradient last layer
        e = self.addBias(e) # Be consistent with the loop
        for i in range(len(self.layer_size)): 
            # Epsilon is vector of errors for neurons only, bias errors from 
            # last layer are removed as they are not considered in the Theta matrix
            e = np.dot(e[:,1:], self.Thetas[-(i+1)]) # Errors * connecting weights
            e = self.sigmoidGrad(self.addBias(self.layer_inputs[-(i+2)]))*e
            self.epsilon_l.append(e)
            # Gradient
            self.delta_l.append(np.dot((e[:,1:]).T, self.activations[-(i+3)]))
        self.epsilon_l.reverse()
        self.delta_l.reverse()
        # Add regularization if defined
        if self.reg_lambda > 0: 
            for i in range(len(self.delta_l)): 
                self.delta_l[i] /= self.training_examples
                reg = np.zeros(self.Thetas[i].shape) # Regularization term
                reg[:,1:] = self.Thetas[i][:,1:]
                self.delta_l[i] += self.reg_lambda*reg*1.0/self.training_examples
    
    def updateWeights(self, learning_rate=None):
        if learning_rate == None: learning_rate = self.learning_rate
        for i in range(len(self.Thetas)): # ~~~ CHANGE FOR APPLY!!! ~~~
            self.Thetas[i] = self.Thetas[i] - self.delta_l[i]*learning_rate
    
    def costFunctionLogistic(self, y):    
        a = self.activations[-1]
        J = (-y*np.log(a) - (1.0-y)*np.log(1.0-a)).sum()
        return J
    
    def costFunctionLinear(self, y):
        a = self.activations[-1]
        m = a.shape[0] # Number of training examples
        J = 1.0/(2*m)*((a-y)**2).sum()
        return J
    
    def cost(self, Y=None):
        if Y == None: Y = self.Y
        y = self.adjustShape(Y, self.outputs)
        y = (y - self.Ymean) / self.Ystd # Scale to network range
        return self.costFunctionLinear(y)
        # ~~~ WHY? ~~~
        if self.output_type == 'logistic': return self.costFunctionLogistic(y)
        elif self.output_type == 'linear': return self.costFunctionLinear(y)
        
    # Measure accuracy in the case of logistic classification
    def accuracy(self, X=None, Y=None):
        if X == None: X = self.X # Use training data if not provided
        if Y == None: Y = self.Y
        Y = self.adjustShape(Y, self.outputs)
        training_examples = X.shape[0]
        Ypred = self.predict(X)
        Ypred_bin = np.zeros(Ypred.shape) # Initialize Y predicted matrix
        true_pos = 0 
        for i,v in enumerate(Ypred.argmax(1)): 
            Ypred_bin[i,v] = 1 # Set to 1 the max of each instance
            if np.all(Ypred_bin[i,:] == Y[i,:]): true_pos += 1
        return true_pos*100.0/training_examples 

    