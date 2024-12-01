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
    ~python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"RTNeural_python";
    ~json_path = ~python_path+/+"MLP_control/data.json";
    RTNeural.saveJSONFile(~dict, ~json_path); //you must install the JSONlib Quark for this to work
)

//train the network using the values in the json file
~out_file = ~python_path+/+"MLP_control/my_mlp_training"; //use a filename without the extension
RTNeural.trainMLP(~json_path, ~out_file); //this will save the pytorch format pt file and rtneural json file

s.boot;

//declare a synthdef
(SynthDef("test_rtneural", {
    RTNeural.kr([MouseX.kr, MouseY.kr], 10, 'mlp', 0).poll;
}).load(s);)

a = Synth("test_rtneural"); //make the synth

//this should load the training and start predicting 10 output values from the MouseX and MouseY inputs
RTNeural.loadModel(a, 'mlp', ~out_file++"_RTNeural.json", true)


// create a 2 input, 13 output synth using RTNeural inference
(
    SynthDef("rtneural_mlp", {
        var out_size = 13;
        var vals = Select.kr(\bypass.kr(1), [RTNeural.kr([\x.kr(0.5), \y.kr(0.5)], out_size, 'mlp', \bypass.kr), \slider_vals.kr((0!out_size))]);
        var osc1, osc2, feed, env;

        vals = Lag.kr(vals);
        
        Out.kr(\vals_bus.kr, vals);

        feed = LocalIn.ar(1);
        osc1 = SinOsc.ar(vals[0].linexp(0,1,1, 3000)+(vals[2].linlin(0,1,0,5000)*feed));

        osc1 = Latch.ar(osc1, Impulse.ar(vals[3].linexp(0,1,200, SampleRate.ir/2)));

        osc1 = RLPF.ar(osc1, vals[4].linexp(0,1,200, SampleRate.ir/2), vals[5]);
        osc1 = (osc1*vals[6].linlin(0,1,0.5,10)).tanh;

        osc2 = SinOsc.ar(vals[7].linexp(0,1,1, 3000)+(vals[8].linlin(0,1,0,5000)*osc1));

        osc2 = Latch.ar(osc2, Impulse.ar(vals[9].linexp(0,1,200, SampleRate.ir/2)));
        
        osc2 = RLPF.ar(osc2, vals[10].linexp(0,1,200, SampleRate.ir/2), vals[11]);
        osc2 = (osc2*vals[12].linlin(0,1,0.5,10)).tanh;

        LocalOut.ar([osc2]);
        
        env = Env.asr(0.1,1,0.01).kr(2, \gate.kr(1));
        Out.ar(0,LeakDC.ar([osc1,osc2],mul: 0.1)*env);
        
    }).load(s);
)

(
    var in_size = 2;
    var out_size = 13;
    var hidden_layers=4;
    var vals_rout, vals_bus = Bus.control(s, out_size);
    var synth = Synth("rtneural_mlp", [\vals_bus, vals_bus]);
    var dict = RTNeural.makeTrainingDict;
    var out_file;
    var reset_dict = {
        var layers;
        dict['in_vals'] = [];
        dict['out_vals'] = [];
        
        layers = 0!hidden_layers;
        ((in_size+2)..out_size).do{|item, i| layers.put(ceil(i * 3 / 11).asInt, item)};
        (layers.size-1).do{|i| layers.put(i, [layers[i], "relu"])};
        layers.put(layers.size-1, [layers.last, "sigmoid"]);
        layers.postln;
        dict['layers_data'] = layers;
        "dictionary reset".postln;
    };

    var bypass = 1;
    var win = Window("RTNeural MultiLayer Perceptron",Rect(0,0,500,300));
    
    var python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"RTNeural_python";

    var msv_array = 0!out_size;
    var xy_array = 0!in_size;

    var msv = MultiSliderView()
    .action_{arg msv;
        var vals = msv.value;
        out_size.do{|i| msv_array[i] = vals[i]};
        synth.set(\slider_vals, msv.value)
    }
    .size_(out_size)
    .elasticMode_(true);
    
    var s2d = Slider2D()
    .action_({ |sl|
        xy_array[0]=sl.x;
        xy_array[1]=sl.y;
        if(bypass==0){
            synth.set(\x, sl.x, \y, sl.y)
        }
    });

    var clear_points_butt = Button()
    .states_([["Clear Points"]])
    .action_({
        reset_dict.value;
    });

    var points_butt = Button()
    .states_([["Add Points"]])
    .action_({
        dict['in_vals'] = dict['in_vals'].add(xy_array.copy);
        dict['out_vals'] = dict['out_vals'].add(msv_array.copy);
        dict['in_vals'].postln;
        dict['out_vals'].postln;
    });

    var load_points_butt = Button()
    .states_([["Load Saved Points"]])
    .action_({
        Dialog.openPanel({ |path|
            var text = File.readAllString(path);
            dict = JSONlib.convertToSC(text);
            dict.postln;
        },{

        }, false, python_path+/+"MLP_control/")
    });
    
    var train_butt = Button(win,Rect(800,20,200,20))
    .states_([["Train", Color.blue, Color.black]])
    .action_({
        Dialog.savePanel({ |path|
            var json_path;
            path = PathName(path);
            path = path.pathOnly++path.fileNameWithoutExtension;
            json_path = path++"_data.json";
            RTNeural.saveJSONFile(dict, json_path); //you must install the JSONlib Quark for this to work
            out_file = path;// without the extension
            RTNeural.trainMLP(json_path, out_file);
        }, {
            "cancelled".postln;
        }, python_path+/+"MLP_control/");
    });

    var load_butt = Button()
    .states_([["load training",Color.green,Color.black]])
    .action_({
        Dialog.openPanel({ |path|
            path.postln;
            RTNeural.loadModel(synth, \mlp, path, 1);
            bypass_butt.valueAction_(1);
        }, {
            "cancelled".postln;
        }, false, python_path+/+"MLP_control/");
    });
    
    var bypass_butt = Button()
    .states_([["Bypass",Color.yellow,Color.black],["Predicting",Color.green,Color.black]])
    .action_({
        arg butt;
        bypass = 1-butt.value;
        synth.set(\bypass, bypass, \slider_vals, msv.value);
    });

    reset_dict.value;
    dict['epochs']=2000;

    win.layout_(
        HLayout(
            msv,
            VLayout(s2d, clear_points_butt, load_points_butt, points_butt, train_butt, load_butt, bypass_butt)
        )
    );
    
    vals_rout = Routine({inf.do{
        if(bypass==0){

            vals_bus.getn(out_size,{|array| 
                {msv.value_(array)}.defer
            });
        };
        0.05.wait;
    }}).play();

    win.front;
    win.onClose = {
        synth.set(\gate, 0);
        vals_rout.stop;
        vals_bus.free;
    };
)

j = "{\"foo\": [42, 56, 75.23]}";

// as event
e = JSONlib.convertToSC(j);
// -> ( 'foo': 42 )

// access values via symbols
e[\foo][0].isInteger;

// as dictionary
d = JSONlib.convertToSC(j, useEvent: false);
// -> Dictionary[ (foo -> 42) ]

// access values via strings
d["foo"];
// -> 42