SynthDef("lil_moogy", {arg freq=100, dur=0.2, vol=0.2;
	var sound, env;

	env = EnvGen.ar(Env([0,1,0], [dur/6,5*dur/6]), 1, doneAction:2);

	sound = Saw.ar(freq)*env*vol;

	sound = MoogFF.ar(sound, freq+(freq*env*2));

	Out.ar(0, sound.dup);
}
).load(s)

//play and record the sequence of ~notes on which we will do the training  
(
    var ps0 = Pseq([0,4,7,11,14,17,14,11,7,4,0,2,4,5,6,8,10,11,12,14,16,17,18,20,22,23], inf).asStream;
    var ps1 = Pxrand([38, 40, 42, 44, 46, 48], inf).asStream;

    ~notes = List.newClear(0);

    {
        inf.do{
            var fund = ps1.next;
            26.do{
                var note = ps0.next+fund;
                ~notes.add(note);
                Synth("lil_moogy",[\freq, note.midicps]);
                0.1.wait;
            }
        }
    }.fork
)

~notes.size;
~notes.do{|item| item.postln};
~notes.pop;  //pop off the notes that don't complete a phrase in the sequence

//write the file containing the entire note sequence so we will have it later
~folder = PathName(RTNeural.filenameSymbol.asString).upMultiDir(2)++"HelpSource/lstm_predict_tutorial/";
~notes.writeArchive(~folder++"notes");

//build the data set and save the json file containing the data set and the training parameters
(
    var memory_size = 10;
    ~in_vals = List.newClear(0);
    ~out_vals = List.newClear(0);

    ~notes.do{|item, i|
        var temp = Array.newClear(memory_size);
        memory_size.do{|j| temp[j] = ~notes[(i+j).wrap(0,~notes.size-1)]};
        ~in_vals.add(temp);
        //make sure that the out_vals are each an array, even though the model is predicting 1 value
        ~out_vals.add([~notes[(i+memory_size).wrap(0,~notes.size-1)]]);
    };

    ~train_data = RTNeural.makeTrainingDict(~in_vals, ~out_vals, [[100,"tanh"]], 1000, 0.001);

    RTNeural.saveJSONFile(~train_data, ~folder++"notes.json");
)

//read the data set back from the saved file
~temp = RTNeural.readJSONFile(~folder++"notes.json")
~temp[\layers_data]

//get the highest note in the data set
//the max output of the softmax lstm output will be 1 - this will represent the highest note of the sequence and the output will scale from 0-1 to 0-highest_note
(
~top = 0;
~notes.do{|item| if(item>~top){~top = item}}
)

~top.postln;

//at this point it is best to do the training on the command line because it will take a couple of minutes
//there is a script in the RTNeural_python folder to do the training - lstm_note_pred/train_lstm_timeseries.py
//run this from the virtual environment you have created inside the RTNeural_python folder
//see RTNeural_python/README.md if you haven't created a virtual environment yet

//the command is:
python <path_to train_lstm_timeseries.py> -f <the json file we just saved> 

(
    //above we chose a memory size of 7, so the 
    var memory_size = 10;
    var max_val = 71;

    //this is a server side method to get the topN indices and topN weights from a softmax probability vector - I realize that this is madness, but it works
    var get_topN = {|softmax, num_vals, num_top=5|
        var max_val, max_index;
        var one_zero = [DC.kr(0)].addAll(Array.fill(num_vals, DC.kr(1)));
        var topN = num_top.collect{
            #max_val, max_index = ArrayMax.kr(softmax);
            one_zero = (1-PanAz.kr(num_vals, DC.kr(1), max_index/num_vals*2, 1, 1, 0));
            softmax = softmax*one_zero;
            [max_index, max_val]
        };
        topN.flop.flat;    
    };

    //this function can be called for each unique RTNeural object with a unique model_id
    var rtneural_next = {|model_id, fb_vals, trig, memory_size, max_val, reset_ins, new_ins, most_or_prob|

        //this chooses between the last N values and a new set of N values
        //since reset_ins is a trigger, it should only take the new_ins once and then revert to the previous N outputs of the LSTM
        var hold_up = SetResetFF.kr(reset_ins, Delay1.kr(trig));
        var in_vals = Select.kr(trig*hold_up, [fb_vals, new_ins]);

        //this will only work in trig_mode 1
        //the input to the network is the last N values of the output or any N values from the training set, where N is time series size the network was trained on
        //the input_size parameter has to be set to the actual input size for the network
        //N needs to be divisible by input_size, so it will not work if the network expects 2 inputs and is given 7, but it will work if it is expecting 1 input and is given memory_size like below
        var predict = RTNeural.kr(in_vals/max_val, max_val+1, model_id, trig_mode:1, trigger: trig, reset: trig, input_size: 1);

        //"predict" is a softmax weighted distribution for the value in a stream
        //topN takes predict and outputs the top N values and their weights
        var topN = get_topN.(predict, max_val+1, 5);

        //chooses the a value from the first half of the topN array with the weights of the second half
        var chosen = TWChoose.kr(trig, topN[0..(5-1)], topN[5..(5*2-1)], 1);

        //either outpus the most probable next value or the weighted choice
        var next_val = Select.kr(most_or_prob, [topN[0], chosen]);

        //removes the oldest value from the last N memory array and adds the next_val to the end of the array
        var feedback_list = Select.kr(trig, [in_vals, [in_vals[1..(memory_size-1)],next_val].flat]);
        Poll.kr(trig, feedback_list);

        feedback_list
    };

    SynthDef("lstm_predict2",{

        //since there can only be one LocalIn/LocalOut pair in the UGen, we have to feedback back the two signals in the same loop
        var note_feedback = LocalIn.kr(memory_size);

        //calculate the next duration, avoiding a duration of 0
        var next_dur = 0.1; 
        
        //choose between triggered OSC messages or the internal Impulse of the UGen 
        var trig = Select.kr(\which_trig.kr(0), [\trigger.tr(0), Impulse.kr(1/next_dur)]);

        //predict the next note and get the last N notes (including the next one)
        var notes_out = rtneural_next.(\notes, note_feedback, trig, memory_size, max_val, \reset_notes.tr(0), \new_notes.kr(0!memory_size), \most_or_prob_notes.kr(0));

        //predict the next dur and get the last 5 durs (including the next one)
        //var dur = rtneural_next.(\bach_dur, note_feedback[5..9], trig, 5, 34, \reset_durs.tr(0), \new_durs.kr(0!5), \most_or_prob_durs.kr(0));

        var env = Env.perc.kr(0, trig);

        var sound = LFTri.ar(notes_out[memory_size-1].midicps)*env*0.1;

        LocalOut.kr(notes_out);

        Out.ar(0, sound.dup);
    }).load(s);
)

(
    a = Synth("lstm_predict2");
    
    RTNeural.loadModel(a, \notes,~folder++"notes_RTNeural.json");

    //RTNeural.loadModel(a, \waits,~folder++"waits_RTNeural.json");
)

~folder = PathName(RTNeural.filenameSymbol.asString).upMultiDir(2)++"HelpSource/lstm_predict_tutorial/";
~notes = Object.readArchive(~folder++"notes");

(
    var notes = ~notes;
    var memory_size = 10;

    var index = notes.size.rand.postln;
    var new_notes = memory_size.collect{|i| notes[(index+i).wrap(0,notes.size-1)]};
    new_notes.postln;
    notes[(index+memory_size).wrap(0,notes.size-1)].postln;

    a.set(\new_notes, new_notes.postln, \reset_notes, 1);
    notes[(index+memory_size).wrap(0,notes.size-1)].postln;
)

//predict a note one trigger at a time
a.set(\trigger, 1, \reset_notes, 0, \reset_durs, 0);

//let the model continuously trigger itself
a.set(\which_trig, 1)

a.set(\most_or_prob_notes, 1);
a.set(\most_or_prob_notes, 0);
