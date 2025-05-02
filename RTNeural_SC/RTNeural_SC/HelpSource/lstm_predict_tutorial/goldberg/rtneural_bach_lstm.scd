//this was a training made in Max using separate note and pitch LSTM networks

//the synthdef used 
(
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

        //this chooses between the last 5 values and a new set of 5 values
        //since reset_ins is a trigger, it should only take the new_ins once and then revert to the previous 5 outputs of the LSTM
        var hold_up = SetResetFF.kr(reset_ins, Delay1.kr(trig));
        var in_vals = Select.kr(trig*hold_up, [fb_vals, new_ins]);

        //this will only work in trig_mode 1
        //the input to the network is the last N values of the output or any N values from the training set, where N is time series size the network was trained on
        //the input_size parameter has to be set to the actual input size for the network
        //N needs to be divisible by input_size, so it will not work if the network expects 2 inputs and is given 7, but it will work if it is expecting 1 input and is given 5 like below
        var predict = RTNeural.kr(in_vals/max_val, max_val+1, model_id, trig_mode:1, trigger: trig, reset: trig, input_size: 1);

        //"predict" is a softmax weighted distribution for the value in a stream
        //topN takes predict and outputs the top N values and their weights
        var topN = get_topN.(predict, max_val+1, memory_size);

        //chooses the a value from the first half of the topN array with the weights of the second half
        var chosen = TWChoose.kr(trig, topN[0..(memory_size-1)], topN[memory_size..(memory_size*2-1)], 1);

        //either outpus the most probable next value or the weighted choice
        var next_val = Select.kr(most_or_prob, [topN[0], chosen]);

        //removes the oldest value from the last N memory array and adds the next_val to the end of the array
        var feedback_list = Select.kr(trig, [in_vals, [in_vals[1..(memory_size-1)],next_val].flat]);

        [next_val, feedback_list]
    };

    SynthDef("lstm_predict",{

        //since there can only be one LocalIn/LocalOut pair in the UGen, we have to feedback back the two signals in the same loop
        //this receives the last 5 notes, last 5 durations (the last value is the next duration)
        var note_dur_feedback = LocalIn.kr(10);

        //calculate the next duration, avoiding a duration of 0
        var next_dur = max(note_dur_feedback[9],1)/10;
        
        //choose between triggered OSC messages or the internal Impulse of the UGen 
        var trig = Select.kr(\which_trig.kr(0), [\trigger.tr(0), Impulse.kr(1/next_dur)]);

        //predict the next note and get the last 5 notes (including the next one)
        var note = rtneural_next.(\bach_notes, note_dur_feedback[0..4], trig, 5, 83, \reset_notes.tr(0), \new_notes.kr(0!5), \most_or_prob_notes.kr(0));

        //predict the next dur and get the last 5 durs (including the next one)
        var dur = rtneural_next.(\bach_dur, note_dur_feedback[5..9], trig, 5, 34, \reset_durs.tr(0), \new_durs.kr(0!5), \most_or_prob_durs.kr(0));

        var env = Env.perc.kr(0, trig);

        var sound = LFTri.ar(note[0].midicps)*env*0.1;

        LocalOut.kr([note[1],dur[1]].flat);

        Out.ar(0, sound);
    }).load(s);
)

(
var path = PathName(RTNeural.filenameSymbol.asString).upMultiDir(2)++"HelpSource/lstm_predict_tutorial/goldberg/";
a = Synth("lstm_predict");

RTNeural.loadModel(a, \bach_notes, path++"bach_notes_RTNeural.json");
RTNeural.loadModel(a, \bach_dur, path++"bach_durs_RTNeural.json");
)


(
    var notes = [79, 79, 81, 79, 81, 83, 81, 79, 78, 76, 74, 67, 66, 67, 69, 67, 66, 67, 69, 67, 69, 67, 69, 67, 69, 67, 69, 67, 66, 67, 69, 67, 66, 67, 66, 64, 64, 62, 74, 74, 76, 74, 76, 77, 76, 74, 72, 71, 69, 79, 78, 76, 78, 79, 78, 81, 79, 78, 76, 74, 72, 72, 81, 72, 71, 67, 66, 66, 67, 66, 67, 71, 71, 73, 71, 73, 74, 74, 73, 71, 69, 67, 71, 76, 79, 81, 79, 78, 79, 81, 79, 81, 79, 81, 79, 81, 79, 81, 79, 78, 79, 79, 78, 76, 74, 73, 71, 73, 74, 73, 74, 73, 74, 73, 76, 81, 79, 78, 76, 74, 69, 71, 72, 71, 69, 67, 66, 64, 74, 73, 74, 76, 74, 73, 71, 69, 79, 71, 73, 73, 74, 76, 74, 73, 73, 74];

    var durs = [16, 16, 2, 2, 8, 4, 8, 4, 4, 8, 25, 2, 2, 12, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 4, 4, 2, 2, 4, 2, 2, 4, 8, 25, 16, 16, 2, 2, 8, 4, 8, 4, 4, 8, 16, 2, 2, 2, 2, 2, 6, 2, 6, 2, 6, 2, 6, 4, 8, 4, 2, 10, 4, 8, 2, 2, 21, 16, 16, 2, 2, 8, 4, 4, 4, 4, 34, 1, 1, 1, 16, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 4, 4, 8, 4, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 8, 8, 4, 4, 4, 4, 8, 18, 2, 4, 4, 4, 4, 4, 8, 4, 14, 2, 4, 4, 4, 4, 4, 8, 16, 8, 4, 6, 2, 2, 2, 8, 16];

    var index = notes.size.rand.postln;
    var new_notes = 5.collect{|i| notes[(index+i).wrap(0,notes.size-1)]};
    //var new_notes = 5.collect{|i| notes.choose}; //when given a random past it will predict a future
    var new_durs = 5.collect{|i| durs[(index+i).wrap(0,durs.size-1)]};

    a.set(\new_notes, new_notes.postln, \reset_notes, 1, \trigger, 1);
    a.set(\new_durs, new_durs.postln, \reset_durs, 1, \trigger, 1);
    notes[(index+5).wrap(0,notes.size-1)].postln;
)


a.set(\which_trig, 1)

a.set(\trigger, 1, \reset_notes, 0, \reset_durs, 0);

a.set(\most_or_prob_notes, 1);
a.set(\most_or_prob_notes, 0);
a.set(\most_or_prob_durs, 1);
a.set(\most_or_prob_durs, 0);