import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        (a, b, c, d) = x.shape
        (w1,w2,w3,w4) = self.W.shape
        y = np.zeros((a, w1, c - w3 + 1, d - w4 + 1))
        conv = np.zeros((w1, c - w3 + 1, d - w4 + 1))
        for i in range(a):
            for j in range(w1):
                temp = view_as_windows(x[i], (w2, w3, w4))
                temp = temp.reshape(c - w3 + 1, d - w4 + 1, -1)

                result = temp.dot(self.W[j].reshape(-1, 1))
                conv[j] = np.squeeze(result, axis=2) + self.b[0][j][0][0]
            y[i] = conv

        out = y

        return out

    def backprop(self, x, dLdy):
        (batch,input_ch,input_w,input_h) = x.shape
        (num_filter,input_ch,Wx,Wy) = self.W.shape
        (a,b,output_w,output_h) = dLdy.shape

        ###############dLdx##############################
        dLdx = np.zeros(x.shape)

        for i in range(batch):
            for j in range(input_ch):
                pad = np.pad(dLdy[i],((0,0),(int(input_w-1+Wx-output_w/2),int(input_w-1+Wx-output_w/2)),(int(input_h-1+Wy-output_h/2),int(input_h-1+Wy-output_h/2))),'constant',constant_values=0)
                flipped_weight = np.flip(self.W[:,j])

                window = view_as_windows(pad,flipped_weight.shape)
                window = window.reshape(input_w,input_h,-1)

                result = window.dot(flipped_weight.reshape(-1,1))
                result = np.squeeze(result,axis=2)

                dLdx[i][j] = result
        ##############dLdW###############################
        dLdW = np.zeros(self.W.shape)
        dLdy_sum = np.sum(dLdy,axis=0)
        x_sum = np.sum(x,axis=0)

        for i in range(num_filter):
            for j in range(input_ch):
                padded_dLdy = np.pad(dLdy_sum[i],((int(Wx+input_w-output_w+1)/2,int(Wx+input_w-output_w+1)/2,int(Wx+input_w-output_w+1)/2),(int(Wy+input_h-output_h+1)/2,int(Wy+input_h-output_h+1)/2)),'constant',constant_values=0)
                flpped_x = np.flip(x_sum[j])

                window2 = view_as_windows(padded_dLdy,flpped_x.shape)
                window2 = window2.reshape(Wx,Wy,-1)

                result2 = window2.dot(flpped_x.reshape(-1,1))
                result2 = np.squeeze(result2,axis=2)

                dLdW[i][j] = result2
        #############dLdb###############################
        dLdb = np.sum(dLdy,axis=0)
        dLdb = np.sum(dLdb,axis=1)
        dLdb = np.sum(dLdb, axis=1)
        dLdb = dLdb.reshape(self.b.shape)

        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        (a, b, c, d) = x.shape
        y = np.zeros((a, b, int((c - self.pool_size) / self.stride + 1), int((d - self.pool_size) / self.stride + 1)))
        pool = np.zeros((b, int((c - self.pool_size) / self.stride + 1), int((d - self.pool_size) / self.stride + 1)))

        for i in range(a):
            temp1 = x[i]
            for j in range(b):
                temp2 = view_as_windows(temp1[j], (self.pool_size, self.pool_size), step=self.stride)
                temp2 = temp2.reshape(int((c - self.pool_size) / self.stride + 1),
                                      int((d - self.pool_size) / self.stride + 1), -1)

                result = np.max(temp2, axis=2)
                pool[j] = result
            y[i] = pool

        out = y

        return out

    def backprop(self, x, dLdy):
        (a, b, c, d) = x.shape
        max_index = np.zeros(
            (a, b, int((c - self.pool_size) / self.stride + 1), int((d - self.pool_size) / self.stride + 1)))
        index_pool = np.zeros(
            (b, int((c - self.pool_size) / self.stride + 1), int((d - self.pool_size) / self.stride + 1)))

        dLdx = np.zeros((a, b, c, d))

        for i in range(a):
            temp1 = x[i]
            for j in range(b):
                temp2 = view_as_windows(temp1[j], (self.pool_size, self.pool_size), step=self.stride)
                temp2 = temp2.reshape(int((c - self.pool_size) / self.stride + 1),
                                      int((d - self.pool_size) / self.stride + 1), -1)

                result = np.argmax(temp2, axis=2)
                index_pool[j] = result
            max_index[i] = index_pool

        for i in range(a):
            for j in range(b):
                for k in range(int((c - self.pool_size) / self.stride + 1)):
                    for l in range(int((d - self.pool_size) / self.stride + 1)):
                        horizon = int(max_index[i][j][k][l] / self.pool_size)
                        vertical = int(max_index[i][j][k][l] % self.pool_size)
                        dLdx[i][j][(k - 1) * self.stride + self.pool_size + horizon][
                            (l - 1) * self.stride + self.pool_size + vertical] = dLdy[i][j][k][l]

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        input = x.reshape(x.shape[0],-1)

        out = input @ self.W.T + self.b

        return out

    def backprop(self,x,dLdy):
        dLdx = dLdy @ self.W
        dLdx = dLdx.reshape(x.shape)

        dLdW = dLdy.T @ x.reshape(x.shape[0],-1)

        dLdb = np.sum(dLdy, axis=0)

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        out = np.maximum(0,x)
        
        return out
    
    def backprop(self, x, dLdy):
        check = (x>0).astype(np.int)

        dLdx = dLdy * check
        
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        exp = np.exp(x)

        sum = np.sum(exp,axis=1)

        (batch,a) = x.shape

        out = np.zeros(x.shape)
        for i in range(batch):
            out[i] = x[i] / sum[i]

        return out

    def backprop(self, x, dLdy):
        exp = np.exp(x)
        sum = np.sum(exp, axis=1)
        square = sum ** 2

        grad = np.zeros(x.shape)
        (a, b) = x.shape
        for i in range(a):
            for j in range(b):
                grad[i][j] = exp[i][j] * (sum[i] - exp[i][j]) / square[i]

        dLdx = dLdy * grad

        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        log = np.log(x) * (-1)

        (batch,) = y.shape

        out = 0

        for i in range(batch) :
            out += log[i][y[i]]

        return out

    def backprop(self, x, y):
        dLdx = np.zeros(x.shape)

        derivative = np.reciprocal(x)
        derivative *= -1

        (batch,) = y.shape
        for i in range(batch):
            dLdx[i][y[i]] = derivative[i][y[i]]

        return dLdx
