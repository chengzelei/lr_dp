import numpy as np
import pylab
import random




def compute_error(b,m,data):

    totalError = 0

    x = data[:,0]
    y = data[:,1]
    totalError = (y-m*x-b)**2
    totalError = np.sum(totalError,axis=0)

    return totalError/float(len(data))

def optimizer(data,starting_b,starting_m,learning_rate,num_iter, c, eps, delta):
    b = starting_b
    m = starting_m

    prev_grad_b_no_noise=0
    prev_grad_b_noise=0
    prev_grad_m_no_noise=0
    prev_grad_m_noise=0
    prev_m=0
    prev_b=0

    xmax=data[0,0]
    for i in range(1,len(data)):
        if data[i,0]>xmax:
           xmax=data[i,0]



    #gradient descent
    for i in range(num_iter):
        #update b and m with the new more accurate b and m by performing
        # thie gradient step
        b,m,prev_b,prev_m,prev_grad_b_no_noise,prev_grad_b_noise,prev_grad_m_no_noise,prev_grad_m_noise =compute_gradient(xmax,b,m,data,learning_rate, c, eps, delta, prev_grad_b_no_noise, prev_grad_b_noise, prev_grad_m_no_noise, prev_grad_m_noise, prev_m, prev_b, i)
        if i%1000==0:
            print 'iter {0}:error={1}'.format(i,compute_error(b,m/xmax,data))
    return [b,m/xmax]

def compute_gradient(xmax,b_current,m_current,data ,learning_rate, c, eps, delta, prev_grad_b_no_noise, prev_grad_b_noise, prev_grad_m_no_noise, prev_grad_m_noise, prev_m, prev_b, iter_num):

    bg = 0
    mg = 0

    b_no_noise=0
    m_no_noise=0
    b_grad_no_noise=0
    m_grad_no_noise=0

    mu=0

    N = float(len(data))
    #Two ways to implement this
    #first way
    for i in range(0,len(data)):
        x = data[i,0]/xmax
        y = data[i,1]

    #
    #     #computing partial derivations of our error function
        b_gtmp= -2*(y-((m_current*x)+b_current))
        m_gtmp= -2 * x * (y-((m_current*x)+b_current))

        #b_gradient += -(2/N)*(y-((m_current*x)+b_current))
       #print b_gradient
       # m_gradient += -(2/N) * x * (y-((m_current*x)+b_current))
        
        w_norm= np.sqrt(np.power(m_gtmp,2)+np.power(b_gtmp,2))
        print w_norm

        if w_norm<=c:
            mg+=m_gtmp
            bg+=b_gtmp
        else:
            mg+=m_gtmp*c/w_norm
            bg+=b_gtmp*c/w_norm

    sensitivity=2*c/N

    sigma=np.sqrt(2*np.log(1.25/delta))*sensitivity/eps

    

    if iter_num==0:
        reuse_coeff=0
        print reuse_coeff

        b_gradient=1/N*(bg+random.gauss(mu,sigma))
        m_gradient=1/N*(mg+random.gauss(mu,sigma))
        prev_b=0
        prev_m=0
        prev_grad_b_no_noise=1/N*bg
        prev_grad_b_noise=b_gradient
        prev_grad_m_no_noise=1/N*mg
        prev_grad_m_noise=m_gradient

    else:
        reuse_coeff=1-np.sqrt(2)*(abs(m_current-prev_m)+abs(b_current-prev_b))/c
        print reuse_coeff

        sigma=np.sqrt((1-reuse_coeff)/(1+reuse_coeff))*sigma
        b_gradient=1/N*bg+reuse_coeff*(prev_grad_b_noise-prev_grad_b_no_noise)+1/N*random.gauss(mu,(1-reuse_coeff)*sigma)
        m_gradient=1/N*mg+reuse_coeff*(prev_grad_m_noise-prev_grad_m_no_noise)+1/N*random.gauss(mu,(1-reuse_coeff)*sigma)
        prev_b=b_current
        prev_m=m_current
        prev_grad_b_no_noise=1/N*bg
        prev_grad_b_noise=b_gradient
        prev_grad_m_no_noise=1/N*mg
        prev_grad_m_noise=m_gradient


    #Vectorization implementation
    #x = data[:,0]
    #y = data[:,1]
    #b_gradient = -(2/N)*(y-m_current*x-b_current)
    #m_gradient = -(2/N)*x*(y-m_current*x-b_current)

    #w_norm= np.sqrt(np.power(m_gradient,2)+np.power(b_gradient,2))

    #print w_norm    
    
    

    #b_gradient = np.sum(b_gradient,axis=0)
    
    #m_gradient = np.sum(m_gradient,axis=0)
        #update our b and m values using out partial derivations

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b,new_m,prev_b,prev_m,prev_grad_b_no_noise,prev_grad_b_noise,prev_grad_m_no_noise,prev_grad_m_noise]


def plot_data(data,b,m):

    #plottting
    x = data[:,0]
    y = data[:,1]
    y_predict = m*x+b
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()


def Linear_regression():
    # get train data
    data =np.loadtxt('data.csv',delimiter=',')

    #define hyperparamters
    #learning_rate is used for update gradient
    #defint the number that will iteration
    # define  y =mx+b
    learning_rate = 0.001
    initial_b =0.0
    initial_m = 0.0
    num_iter = 1000

    clipping_bound=1

    eps=1
    delta=0.01

    #train model
    #print b m error
    print 'initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n'\
        .format(initial_b,initial_m,compute_error(initial_b,initial_m,data))

    #optimizing b and m
    [b ,m] = optimizer(data,initial_b,initial_m,learning_rate,num_iter,clipping_bound, eps, delta)

    #print final b m error
    print 'final formula parmaters:\n b = {1}\n m={2}\n error of end = {3} \n'.format(num_iter,b,m,compute_error(b,m,data))

    #plot result
    plot_data(data,b,m)

if __name__ =='__main__':

    Linear_regression()
