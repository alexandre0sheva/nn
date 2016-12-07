var webpack = require('webpack');

module.exports = {
    entry: "./nn.js",
    output: {
        path: __dirname + '/public/build/',
        publicPath: "build/",
        filename: "main.js"
    },
    module: {
        loaders: [
            {
                test: /\.js$/,
                loader: "babel",
                exclude: [/node_modules/, /public/]
            }
        ]
    }
}
