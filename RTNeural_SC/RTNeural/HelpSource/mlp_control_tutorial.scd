//make a default dictionary used for training the network
//normally, we will want to replace the arrays stored under the in_vals and out_vals keys
~dict = RTNeural.makeTrainingDict;

//see the values stored in the keys - the python script will use these keys to create and train the network
//keys are:
//layers_data - the shape of the neural network
//in_vals - the input values to be trained on
//out_vals - the output values to be trained on
//epochs - how many epochs in the training loop
//learn_rate - the initial learning rate of the training
~dict.keysDo{|key| [key, ~dict[key]].postln}; nil;


(
    //save the json file (can go anywhere)
    ~python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"python";
    ~json_path = ~python_path+/+"MLP_control/data.json";
    // RTNeural.saveJSONFile(~dict, ~json_path);
)

//train the network using the values in the json file
RTNeural.trainMLP(~json_path);
~pt_file = "/Users/spluta1/Library/Application Support/SuperCollider/Extensions/MyPlugins/RTNeuralUGen/RTNeural/python/MLP_control/mlp_training.pt"
RTNeural.convertToRTNeural(~pt_file)


// 2. the synth that we'll be contorlling (has 10 control parameters, expects them to be between 0 and 1)
(
    SynthDef("rtneural_mlp", {
        var val = RTNeural.kr([\x.kr(0.5), \y.kr(0.5)], 10, 'mlp');
        var osc1, osc2, feed;
        feed = LocalIn.ar(1);
        osc1 = SinOsc.ar(val[0].linexp(0,1,1, 3000)+(val[2].linlin(0,1,0,5000)*feed));
        osc2 = SinOsc.ar(val[1].linexp(0,1,1, 3000)+(val[3].linlin(0,1,0,5000)*osc1));
        Out.ar(0,LeakDC.ar([osc1,osc2],mul: 0.1));
        LocalOut.ar([osc2]);
    }).load(s);
)

~synth = Synth("rtneural_mlp");
s.scope

~python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"python";
RTNeural.loadModel(~synth, \mlp, ~python_path+/+"MLP_control/mlp_RTNeural.json");

    // 3. a gui for creating input-output example pairs, then training, and then making predicitons.
(
    ~counter = 0;
    ~predicting = false;
    ~prediction_buf = Buffer.alloc(s,10);
    ~win = Window("MLP Regressor",Rect(0,0,1000,400));
    
    ~multisliderview = MultiSliderView(~win,Rect(0,0,400,400))
    .size_(10)
    .elasticMode_(true)
    .action_({
        arg msv;
        ~y_buf.setn(0,msv.value);
    });
    
    Slider2D(~win,Rect(400,0,400,400))
    .action_({
        arg s2d;
        [s2d.x,s2d.y].postln;
        ~x_buf.setn(0,[s2d.x,s2d.y]); // put the x position directly into the buffer
        if(~predicting,{
            ~nn.predictPoint(~x_buf,~y_buf,{
    
                // the synth actually reads directly out of the ~y_buf, but here we get it back to the language so we can display
                // it in the multislider
                ~y_buf.getn(0,10,{
                    arg prediction_values;
                    {~multisliderview.value_(prediction_values)}.defer;
                });
            });
        });
    });
    
    Button(~win,Rect(800,0,200,20))
    .states_([["Add Points"]])
    .action_({
        arg but;
        var id = "example-%".format(~counter);
        ~ds_input.addPoint(id,~x_buf);
        ~ds_output.addPoint(id,~y_buf);
        ~counter = ~counter + 1;
    
        ~ds_input.print;
        ~ds_output.print;
    });
    
    Button(~win,Rect(800,20,200,20))
    .states_([["Train"]])
    .action_({
        arg but;
        ~nn.fit(~ds_input,~ds_output,{
            arg loss;
            "loss: %".format(loss).postln;
        });
    });
    
    Button(~win,Rect(800,40,200,20))
    .states_([["Not Predicting",Color.yellow,Color.black],["Is Predicting",Color.green,Color.black]])
    .action_({
        arg but;
        ~predicting = but.value.asBoolean;
    });
    
    ~win.front;
)