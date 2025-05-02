~dict = RTNeural.makeTrainingDict;

//make sure to replace the layers_data with the correct shape
~dict[\layers_data] = [[100, "relu"], [100,"relu"], [1,"sigmoid"]]

~dict.keysDo{|key| [key, ~dict[key]].postln}; nil;

//fill the in_vals and out_vals with arrays of inputs and outputs
(
    var in_vals = List.newClear(0);
    var out_vals = List.newClear(0);

    5000.do{
        var temp = [11.rand, 11.rand];
        in_vals.add(temp/10);
        out_vals.add([temp.sum/20]);
    };

    ~dict[\in_vals] = in_vals;
    ~dict[\out_vals] = out_vals;
)


//set the path to the json file
~folder = PathName(RTNeural.filenameSymbol.asString).upMultiDir(2)++"HelpSource/mlp_adding_tutorial/";

//this requres the JSONlib Quark - if you don't have it, run 'Quarks.gui' and install it
RTNeural.saveJSONFile(~dict, ~folder++"mlp_adding.json";)

//set the path to where your RTNeural_python directory is...not where mine is
~python_path = "/Users/spluta1/Desktop/RTNeural_python";

RTNeural.trainMLP(~folder++"mlp_adding.json", nil, ~python_path)

//make our SynthDef

SynthDef("mlp_adder", {
    var answer = (RTNeural.kr(\in_vals.kr([0,0]), 1, \adder)*20);
    SendReply.kr(\trig.tr(0), '/answer', answer, -1);
    nil
}
).load(s)

//make a synth
(
a = Synth("mlp_adder");
o = OSCFunc({ |msg| msg.postln }, '/answer');
)
//load the model
RTNeural.loadModel(a, \adder, ~folder++"mlp_adding_RTNeural.json")

//set the values and get the results
a.set(\in_vals, [11.0.rand, 11.0.rand].postln/10, \trig, 1)