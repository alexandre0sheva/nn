var R = require('ramda');
void 0; //to not bloat the output
var random = require('seed-random')(1337);
var data = [
    {input: [0, 0], output: 0},
    {input: [1, 0], output: 1},
    {input: [0, 1], output: 1},
    {input: [1, 1], output: 0},
];
var activation_sigmoid = x => 1 / (1 + Math.exp(-x));
var derivative_sigmoid = x => {
    const fx = activation_sigmoid(x);
    return fx * (1 - fx);
};

var weights = {
    i1_h1: random(),
    i2_h1: random(),
    bias_h1: random(),
    i1_h2: random(),
    i2_h2: random(),
    bias_h2: random(),
    h1_o1: random(),
    h2_o1: random(),
    bias_o1: random(),
};

function nn(i1, i2) {
    var h1_input =
        weights.i1_h1 * i1 +
        weights.i2_h1 * i2 +
        weights.bias_h1;
    var h1 = activation_sigmoid(h1_input);

    var h2_input =
        weights.i1_h2 * i1 +
        weights.i2_h2 * i2 +
        weights.bias_h2;
    var h2 = activation_sigmoid(h2_input);


    var o1_input =
        weights.h1_o1 * h1 +
        weights.h2_o1 * h2 +
        weights.bias_o1;

    var o1 = activation_sigmoid(o1_input);

    return o1;
}

var calculateResults = () =>
    R.mean(data.map(({input: [i1, i2], output: y}) => Math.pow(y - nn(i1, i2), 2)));

var outputResults = () =>
    data.forEach(({input: [i1, i2], output: y}) =>
        console.log(`${i1} XOR ${i2} => ${nn(i1, i2)} (expected ${y})`));

outputResults()

var train = () => {
    const weight_deltas = {
        i1_h1: 0,
        i2_h1: 0,
        bias_h1: 0,
        i1_h2: 0,
        i2_h2: 0,
        bias_h2: 0,
        h1_o1: 0,
        h2_o1: 0,
        bias_o1: 0,
    };

    for (var {input: [i1, i2], output} of data) {
        //this part is 100% identic to forward pass function
        var h1_input =
            weights.i1_h1 * i1 +
            weights.i2_h1 * i2 +
            weights.bias_h1;
        var h1 = activation_sigmoid(h1_input);

        var h2_input =
            weights.i1_h2 * i1 +
            weights.i2_h2 * i2 +
            weights.bias_h2;
        var h2 = activation_sigmoid(h2_input);


        var o1_input =
            weights.h1_o1 * h1 +
            weights.h2_o1 * h2 +
            weights.bias_o1;

        var o1 = activation_sigmoid(o1_input);

        //learning starts here:
        // we calculate our delta
        var delta = output - o1;
        //then we calculate our derivative (and throwing away "2 * " as we can multiply it later)
        var o1_delta = delta * derivative_sigmoid(o1_input);

        //and for our equatation w1 * h1 + w2 * h2 we're trying to alter weights first

        weight_deltas.h1_o1 += h1 * o1_delta;
        weight_deltas.h2_o1 += h2 * o1_delta;
        weight_deltas.bias_o1 += o1_delta;

        //and then we're trying to alter our h1 and h2.
        //but we cannot alter them directly, as they are functions of other weights too
        //so we need to alter their weights by same approach

        var h1_delta = o1_delta * derivative_sigmoid(h1_input);
        var h2_delta = o1_delta * derivative_sigmoid(h2_input);

        weight_deltas.i1_h1 += i1 * h1_delta;
        weight_deltas.i2_h1 += i2 * h1_delta;
        weight_deltas.bias_h1 += h1_delta;

        weight_deltas.i1_h2 += i1 * h2_delta;
        weight_deltas.i2_h2 += i2 * h2_delta;
        weight_deltas.bias_h2 += h2_delta;
    }

    return weight_deltas;
}

var applyTrainUpdate = (weight_deltas = train()) =>
    Object.keys(weights).forEach(key =>
        weights[key] += weight_deltas[key]);

console.log('start train');
applyTrainUpdate();
outputResults();
calculateResults();

console.log('start train 100');
R.times(() => applyTrainUpdate(), 100)
outputResults();
calculateResults();

console.log('start train 1000');
R.times(() => applyTrainUpdate(), 1000)
outputResults();
calculateResults();

console.log('start train 2000');
R.times(() => applyTrainUpdate(), 2000)
outputResults();
calculateResults();
