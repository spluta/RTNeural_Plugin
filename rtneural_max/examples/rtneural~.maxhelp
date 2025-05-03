{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 100.0, 100.0, 723.0, 726.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-6",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 610.0, 278.0, 108.0, 22.0 ],
					"text" : "rtneural~_gru_saw"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 608.5, 483.0, 134.0, 22.0 ],
					"text" : "test_lstm_max_vs_msp"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 608.5, 410.0, 143.0, 22.0 ],
					"text" : "rtneural~_mlp_wavetable"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 610.0, 339.0, 160.0, 22.0 ],
					"text" : "rtneural~_mlp_trigger_mode"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 610.0, 204.0, 149.0, 22.0 ],
					"text" : "rtneural~_lstm_two_inputs"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 610.0, 138.0, 140.0, 22.0 ],
					"text" : "rtneural~_lstm_distortion"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"linecount" : 10,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 58.0, 60.0, 363.0, 141.0 ],
					"text" : "rtneural~ is the audio rate max rtneural external.\n\nthis object can load any neural network model saved in rtneural format for inference.\n\nif the patchers on the right are not available, be sure to add the rtneural_max folder to your path\n\neach patch has independent dsp to avoid overtaxing the cpu, so be sure to turn on the dsp in the individual patches"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"linecount" : 27,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 584.0, 93.0, 489.0, 382.0 ],
					"text" : "use a single input LSTM or GRU as a distortion\n\n\n\n\n\nuse a two channel LSTM or GRU as a distortion\n\n\n\n\na GRU model trained to convert a sine wave into an anti-aliased sawtooth wave\n\n\n\n\nrun multi layer perceptron inference at audio rate at triggered moments\n\n\n\n\nuse a multi layer perceptron as a wavetable lookup\n\n\n\n\ncompare the output of the max lstm note prediction and the msp note prediction\n"
				}

			}
 ],
		"lines" : [  ],
		"parameters" : 		{
			"obj-5::obj-32" : [ "level", "level", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "div_by_maxval.js",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "drumLoop.aif",
				"bootpath" : "C74:/media/msp",
				"type" : "AIFF",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "rtneural~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "rtneural~_gru_saw.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/rnn_distortion",
				"patcherrelativepath" : "./rnn_distortion",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural~_lstm_distortion.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/rnn_distortion",
				"patcherrelativepath" : "./rnn_distortion",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural~_lstm_two_inputs.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/rnn_distortion",
				"patcherrelativepath" : "./rnn_distortion",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural~_mlp_trigger_mode.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/mlp",
				"patcherrelativepath" : "./mlp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural~_mlp_wavetable.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/mlp",
				"patcherrelativepath" : "./mlp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "test_lstm_max_vs_msp.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "JSON",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
