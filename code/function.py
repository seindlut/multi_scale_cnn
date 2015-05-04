import theano
import numpy

x = theano.tensor.matrix('x')   # symbolic variable
cost = x ** 2                   # relationship

weight1 = theano.shared(numpy.asarray([[1,2],[3,4]], dtype=theano.config.floatX), borrow=True)                      # weight1 is a shared variable

weight2 = theano.shared(numpy.asarray([[5,6],[7,8]], dtype=theano.config.floatX), borrow=True)                      # weight2 is a shared variable

weight = [weight1] + [weight2]  # weight concatenated

shared_x = theano.shared(numpy.asarray([[1,2,3,4],[5,6,7,8]], dtype=theano.config.floatX), borrow=True)             # shared_x is a shared variable

grad1 = theano.shared(numpy.asarray([[0,-1],[1,2]], dtype=theano.config.floatX), borrow=True)                       # grads is a shared variable

grad2 = theano.shared(numpy.asarray([[-1,0],[0,-2]], dtype=theano.config.floatX), borrow=True)

grad = [grad1] + [grad2]
updates = [(weight1, weight1 - 0.1 * grad1),
           (weight2, weight2 - 0.2 * grad2)]
#updates = [(weight_i, weight_i - grad_i) for weight_i, grad_i in zip(weight, grad)]
                                # update ruls

func = theano.function([], cost, updates=updates, givens={x:shared_x})

print "weight1"
print weight1.get_value()
print "weight2"
print weight2.get_value()
print "function called"
func()
print "weight1"
print weight1.get_value()
print "weight2"
print weight2.get_value()

