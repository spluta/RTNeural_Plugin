RTNeural : MultiOutUGen {
	var <>id, <>desc;
	*ar { |input_array, num_outputs, id, bypass=0, sample_rate=(-1)|
        ^this.multiNewList(['audio', num_outputs, id, bypass, sample_rate] ++ input_array.asArray)
	}

	*kr { |input_array, num_outputs, id, bypass=0, sample_rate=(-1)|
        ^this.multiNewList(['control', num_outputs, id, bypass, sample_rate] ++ input_array.asArray)
	}

	init { arg argNumOutChannels, argID ... theInputs;
		this.id = argID;
		inputs = theInputs;
		^this.initOutputs(argNumOutChannels, rate);
	}

	*loadModel {|synth, id, path, verbose = true|
		//get the index from SynthDescLib
		var defName = synth.defName.asSymbol;
		var synthIndex = SynthDescLib.global[defName];
		
		if (synthIndex != nil) {
			synthIndex=synthIndex.metadata()[defName][id.asSymbol]['index'];
		}{
			SynthDescLib.read(SynthDef.synthDefDir+/+defName.asString++".scsyndef");
			synthIndex = SynthDescLib.global[defName].metadata()[defName][id.asSymbol]['index'];
		};

		if (synthIndex == nil){
			"SynthDef has no metadata.\n".error;
		};

		//no multichannel expansion possible
		//synthIndex.do{|index|
		synth.server.sendMsg('/u_cmd', synth.nodeID, synthIndex, 'load_model', path, verbose);
		//}
	}

	*makeTrainingDict {arg in_vals = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], out_vals = [[0.5875146667089205, 0.8485457514926228, 0.2327253852912824, 0.07685113193603721, 0.4898428344560041, 0.8606441449537087, 0.3584816718414401, 0.6643172227394102, 0.27106557261260966, 0.9120306316810372], [0.3166591905634991, 0.048534802525801335, 0.73056053455773, 0.3496541632686534, 0.8626464600215307, 0.023862736866254286, 0.48475295327051926, 0.3885668471363576, 0.8134163028463399, 0.255052680831537], [0.4518235200263505, 0.4137559403502248, 0.5022822296817467, 0.17024503344045772, 0.008492093143773305, 0.5909011245334566, 0.48922326094601354, 0.7477378675068113, 0.7663491755528566, 0.27479012583296736], [0.18749091234448634, 0.19019111860337168, 0.8523199754755959, 0.33290852492301404, 0.6087125492599167, 0.3314290451707914, 0.9677761640407265, 0.9881061220527863, 0.820468733508028, 0.47150751034615035]], layers_data = [[3, "relu"], [5, "relu"], [7, "relu"], [9, "relu"], [10, "sigmoid"]], epochs = 5000, learn_rate = 0.001;
		var a = Dictionary();
		a.put(\in_vals, in_vals);
		a.put(\out_vals, out_vals);
		a.put(\layers_data, layers_data);
		a.put(\epochs, epochs);
		a.put(\learn_rate, learn_rate);

		^a
	}

	*trainMLP{
		arg in_file = "MLP_control/data.json", 
		out_file, 
		python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"RTNeural_python",
		action = {};

		var text;
		if (out_file!=nil){
			text = "cd "++python_path.quote++"; source venv/bin/activate; python MLP_control/mlp_control_train_convert.py -f"+in_file.quote+"-o"+out_file.quote;
		}{
			text = "cd "++python_path.quote++"; source venv/bin/activate; python MLP_control/mlp_control_train_convert.py -f"+in_file.quote;
			out_file = PathName(in_file)
		};

		text.unixCmd{ |res, pid| 
			//after it is trained, the model needs to be converted to RTNeural format
			"\n \n ready for inference!".postln;
		};
		"wait for the done message. it'll take a second or longer depending on epochs.".postln;
	}

	*convertToRTNeural {
		arg pt_file = "MLP_control/mlp_control_training.pt", 
		rt_neural_file, 
		python_path = PathName(RTNeural.filenameSymbol.asString).pathOnly++"RTNeural_python";

		var text = "cd "++python_path.quote++"; source venv/bin/activate; python MLP_control/torch_to_RTNeural_MLP.py -f"+pt_file.quote;
		if (rt_neural_file!=nil){
			text = "cd "++python_path.quote++"; source venv/bin/activate; python MLP_control/torch_to_RTNeural_MLP.py -f"+pt_file.quote+"-o"+rt_neural_file.quote;
		};
		text.postln;
		text.unixCmd{ |res, pid| 
			//after it is trained, the model needs to be converted to RTNeural format
			"ready for inference".postln;
		};
		"wait for the conversion message".postln;
	}

	*saveJSONFile {arg dict, path;
		var j = JSONlib.convertToJSON(dict); //you must install the JSONlib Quark
		var f = File(path, "w");
		f.write(j);
		f.close;	
	}

	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

	synthIndex_ { arg index;
		super.synthIndex_(index); // !
		// update metadata (ignored if reconstructing from disk)
		this.desc.notNil.if { this.desc.index = index; }
	}

	optimizeGraph {
		var metadata;

		// This is called once per UGen during SynthDef construction!
		
		// For older SC versions, where metadata might be 'nil'
		this.synthDef.metadata ?? { this.synthDef.metadata = () };
		
		metadata = this.synthDef.metadata[this.synthDef.name];
		if (metadata == nil) {
			// Add RTNeural metadata entry if needed:
			metadata = ();
			this.synthDef.metadata[this.synthDef.name] = metadata;
			this.desc = ();
			this.desc[\index] = this.synthIndex;

		}{
			//if the metadata already existed, that means there are multiple UGens with the same id
			
			this.desc = ();
			if (metadata[this.id.asSymbol]==nil){
				//if the id info is not there, it is an additional id
				this.desc[\index] = this.synthIndex;
			}{
				Error("Each RTNeural instance in a Synth must have a unique ID.").throw;
				//if the symbol is there, it is probably multichannel expansion
				//so we load all the indexes into an array so we can set them all at once
				//this.desc[\index] = (metadata[this.id.asSymbol][\index].add(this.synthIndex));
			};
		};

		this.id.notNil.if {
			metadata.put(this.id, this.desc);
		}{
			Error("Each RTNeural instance in a Synth must have a unique ID.").throw;
		};
	}
}
