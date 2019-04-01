using LinearAlgebra
using Random
using MLDatasets

# Network object
mutable struct NeuralNetwork
    weights::Array{Array{Float64,2},1}
    biases::Array{Array{Float64,1},1}
end


# Function creates a network of any size. The number of neurons
# in each layer are passed in a tuple
function createNetwork(layers::Tuple)
    network = NeuralNetwork([],[])
    for i in 1:length(layers)-1
        weightMatrix = randn(layers[i+1],layers[i])
        push!(network.weights,weightMatrix)
        biasVector = randn(layers[i+1])
        push!(network.biases,biasVector)
    end
    return network
 end


# Function takes a network and feeds an input vector through
# to return lists of activations and z = (W*a + b)
function feedForward(network::NeuralNetwork, x::Array)
    activations = [x]
    zList = []

    for (W,b) in zip(network.weights, network.biases)
        z = W*x + b
        push!(zList, z)
        x = sigmoid.(z)
        push!(activations, x)
    end

    return activations, zList
end

# Function passes single input through network and returns answer
function ffInput(network::NeuralNetwork, x::Array)
    for (W,b) in zip(network.weights, network.biases)
        x = sigmoid.(W*x + b)
    end
    return x
end

# Function takes a network, input, and label to compute gradients
# for the network's weights and biases and returns these gradients 
# in lists
function backPropagation(network::NeuralNetwork, input::Array, label::Int64)
    label = labelToVector(label)
    nabla_w = [] # Array to hold weight gradients
    nabla_b = [] # Array to hold bias gradients
    aList, zList = feedForward(network, input)
    delta = (aList[end] - label) .* sigmoidPrime.(zList[end])
    pushfirst!(nabla_b, delta)
    pushfirst!(nabla_w, delta * aList[end - 1]')

    for i in 0:length(network.weights)-2
        delta = (net.weights[end - i]' * delta) .* sigmoidPrime.(zList[end - i - 1])
        pushfirst!(nabla_b, delta)
        pushfirst!(nabla_w, delta * aList[end - i - 2]')
    end
    
    return nabla_b, nabla_w
end

#Function does the same as above with cross entropy instead of quadratic cost
function backPropagation2(network::NeuralNetwork, input::Array, label::Int64)
    label = labelToVector(label)
    nabla_w = [] # Array to hold weight gradients
    nabla_b = [] # Array to hold bias gradients
    aList, zList = feedForward(network, input)
    delta = aList[end] - label
    pushfirst!(nabla_b, delta)
    #wDelta = delta * aList[end - 1]'
    pushfirst!(nabla_w, delta * aList[end - 1]')

    for i in 0:length(network.weights)-2
        delta = (net.weights[end - i]' * delta) .* sigmoidPrime.(zList[end - i - 1])
        pushfirst!(nabla_b, delta)
        #wDelta = delta * aList[end - i - 2]'
        pushfirst!(nabla_w, delta * aList[end - i - 2]')
    end
    
    return nabla_b, nabla_w
end

function updateMiniBatch!(network::NeuralNetwork, miniBatch::Array, eta::Number)
    m = length(miniBatch)
    nablaB = [zero(b) for b in network.biases]
    nablaW = [zero(w) for w in network.weights]
    for (input, label) in miniBatch
        deltaB, deltaW = backPropagation(network, input, label)
        nablaB = [nb+dnb for (nb, dnb) in zip(nablaB, deltaB)]
        nablaW = [nw+dnw for (nw, dnw) in zip(nablaW, deltaW)]
    end
    network.weights = [W-(eta/m)*nW for (W, nW) in zip(network.weights,nablaW)]
    network.biases = [b-(eta/m)*nb for (b, nb) in zip(network.biases,nablaB)]
end


function updateMiniBatch2!(network::NeuralNetwork, miniBatch::Array, eta::Number)
    m = length(miniBatch)
    nablaB = [zero(b) for b in network.biases]
    nablaW = [zero(w) for w in network.weights]
    for (input, label) in miniBatch
        deltaB, deltaW = backPropagation2(network, input, label)
        nablaB = [nb+dnb for (nb, dnb) in zip(nablaB, deltaB)]
        nablaW = [nw+dnw for (nw, dnw) in zip(nablaW, deltaW)]
    end
    network.weights = [W-(eta/m)*nW for (W, nW) in zip(network.weights,nablaW)]
    network.biases = [b-(eta/m)*nb for (b, nb) in zip(network.biases,nablaB)]
end

# Function performs stochastic gradient descent
function SGD!(network::NeuralNetwork, trainingData::Array, testData::Array, epochs::Number, miniBatchSize::Number, eta::Number)
    n = length(trainingData)
    nTest = length(testData)
    for i in 1:epochs
        shuffle!(trainingData)
        miniBatches = [trainingData[k:k+miniBatchSize-1] for k in 1:miniBatchSize:n]
        for mb in miniBatches
            updateMiniBatch!(network,mb,eta)
        end
        numCorrect = evaluate(network,testData)
        println("Epoch $i: $numCorrect/$nTest")
    end
end


# Same as above with cross entropy
function SGD2!(network::NeuralNetwork, trainingData::Array, testData::Array, epochs::Number, miniBatchSize::Number, eta::Number)
    n = length(trainingData)
    nTest = length(testData)
    for i in 1:epochs
        shuffle!(trainingData)
        miniBatches = [trainingData[k:k+miniBatchSize-1] for k in 1:miniBatchSize:n]
        for mb in miniBatches
            updateMiniBatch2!(network,mb,eta)
        end
        numCorrect = evaluate(network,testData)
        println("Epoch $i: $numCorrect/$nTest")
    end
end

sigmoid(x) = 1/(1 + exp(-x))
sigmoidPrime(x) = sigmoid(x)*(1-sigmoid(x))

# Function returns the number of test inputs for which the network
# outputs the correct result
function evaluate(network::NeuralNetwork, testData::Array)::Number
    testResults = [(argmax(ffInput(network,input))-1,label) for (input,label) in testData]
    return sum(Int64(x==y) for (x,y) in testResults)
end


# function takes an image label number and creates a 10x1 vector 
# with a 1 in the position of the number
function labelToVector(label::Int64)
    labelVector = zeros(10)
    labelVector[label + 1] = 1
    return labelVector
end

# Function converts mnist data into workable format
function mnistConverter(inputs,labels)
    vectors = [convert(Array{Float64,1},vec(inputs[:, :, i])) for i in 1:size(inputs,3)]
    return collect(zip(vectors,convert(Array{Int64,1},labels)))
end